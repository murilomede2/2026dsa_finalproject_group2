library(tidymodels)   # Core framework for modeling (includes recipes, workflows, parsnip, etc.)
library(finetune)     # Additional tuning strategies (e.g., racing, ANOVA-based tuning)
library(vip)          # For plotting variable importance from fitted models
library(xgboost)      # XGBoost implementation in R
library(ranger)       # Fast implementation of Random Forests
library(tidyverse)    # Data wrangling and visualization
library(doParallel)   # For parallel computing (useful during resampling/tuning)
library(caret)        # Other great library for Machine Learning 
library(here)
library(readr)
library(dplyr)


corn_training <- read.csv(here("data", "corn_training.csv"))

set.seed(9678965) # Setting seed to get reproducible results 

corn_split_xgb <- initial_split(corn_training, 
  prop = .7,
  strata = yield_mg_ha) # proportion of split same as previous codes

corn_split_xgb

corn_train_xgb <- training(corn_split_xgb)  # 70% of data

corn_train_xgb #This is your training data frame


corn_test_xgb <- testing(corn_split_xgb)

corn_test_xgb

 density_plot_xgb <- ggplot() +
  geom_density(data = corn_train_xgb, 
               aes(x = yield_mg_ha),
               color = "red") +
  geom_density(data = corn_test_xgb, 
               aes(x = yield_mg_ha),
               color = "blue") 

cat(paste0("Saving...... density plot of train and test data"))

ggsave(plot = density_plot_xgb,
       path = here("output", "png"),
       filename = "density_plot_test_train_xgb.png",
       height = 6,
       width = 9,
       dpi = 600)


# Create recipe for data preprocessing
corn_recipe_xgb <- recipe(yield_mg_ha ~ ., data = corn_train_xgb) %>% 
  step_impute_median(all_numeric_predictors()) %>%   #changing NAs values to median values
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

corn_recipe_xgb

# Prep the recipe to estimate any required statistics
corn_prep_xgb <- corn_recipe_xgb %>% 
  prep()

# Examine preprocessing steps
corn_prep_xgb


xgb_spec <- #Specifying XgBoost as our model type, asking to tune the hyperparameters
  boost_tree(
   # Total number of boosting iterations
    trees = tune(), 
         # Maximum depth of each tree
    tree_depth = tune(),
             # Minimum samples required to split a node
    min_n = tune(),
        # Step size shrinkage for each boosting step
    learn_rate = tune()) %>%
        #specify engine 
  set_engine("xgboost") %>%
       # Set to mode
  set_mode("regression")

xgb_spec


set.seed(34549) #34549

resampling_foldcv_xgb <- vfold_cv(corn_train_xgb, # Create 5-fold cross-validation resampling object from training data
                              v = 10)


xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  learn_rate(),
  trees(),
  size = 50)

xgb_grid

 grid_plot_xgb <- ggplot(data = xgb_grid,
       aes(x = tree_depth, 
           y = min_n)) +
  geom_point(aes(color = factor(learn_rate),
                 size = trees),
             alpha = .5,
             show.legend = FALSE)

cat(paste0("Saving...... grid plot from test data"))

ggsave(plot = grid_plot_xgb,
       path = here("output", "png"),
       filename = "grid_plot_xgb.png",
       height = 6,
       width = 9,
       dpi = 600)


#Detecting cores in Sapelo:

n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))

if (is.na(n_cores) || n_cores < 1) {
  n_cores <- parallel::detectCores() -1
  
} 

#Start the cluster
cl <- makePSOCKcluster(n_cores)

registerDoParallel(cl)

cat(paste0("\nFound and registered", n_cores, "cores to work with\n"))


set.seed(6576)

# Creating the list of cross validation techniques so we can loop through them
cv_list_xgb <- list(vfold = resampling_foldcv_xgb)

#Create a empty list to store the results from the loop
results_xgb <- list()

#Create the loop
for (i in seq_along(cv_list_xgb)) {
  
  name_xgb <- names(cv_list_xgb)[i]
  
  results_xgb[[name_xgb]] <- tune_race_anova(object = xgb_spec,
                                     preprocessor = corn_recipe_xgb,
                                     resamples = cv_list_xgb[[i]],
                                     grid = xgb_grid,
                                     control = control_race(save_pred = TRUE,
                                                            parallel_over = "everything"))
}

stopCluster(cl)



#Create a data frame structure from the list obtained after running the loop

results_df_xgb <- tibble(method = names(results_xgb),
                         diff_cv = results_xgb) %>%

#Collecting the metrices for each cross validation techniques using map function

mutate(metrices = map2(diff_cv, method,
                       ~.x %>%
                         collect_metrics() %>%
                         mutate(method = .y, .before = "trees")))

results_df_xgb$metrices[[1]]

results_df_xgb

#Bind all the metrices together to select the best performing one

all_metrices_xgb <- do.call(bind_rows,
                            results_df_xgb$metrices)



#Automating to pull the best method out of 3 we ran

best_method_xgb <- all_metrices_xgb %>%
  filter(.metric =="rmse") %>%
  slice_min(mean, n = 1) %>%
  pull(method)

#Getting the metrice (hyperparameter values of the best performing cross validation)

best_cv_object_xgb <- results_df_xgb %>%
  filter(method == best_method_xgb) %>%
  pull(diff_cv) %>%
  first()



# Based on lowest RMSE
best_rmse_xgb <- best_cv_object_xgb %>% 
  select_best(metric = "rmse")%>% 
  mutate(source = "best_rmse")

best_rmse_xgb


# Based on greatest R2
best_r2_xgb <- best_cv_object_xgb %>% 
  select_best(metric = "rsq")%>% 
  mutate(source = "best_r2")

best_r2_xgb



best_rmse_xgb %>% 
  bind_rows(best_rmse_xgb, 
            best_r2_xgb) %>%
  dplyr::select(source, everything())


final_spec_xgb <- boost_tree(
  trees = best_rmse_xgb$trees,           # Number of boosting rounds (trees)
  tree_depth = best_rmse_xgb$tree_depth, # Maximum depth of each tree
  min_n = best_rmse_xgb$min_n,           # Minimum number of samples to split a node
  learn_rate = best_rmse_xgb$learn_rate  # Learning rate (step size shrinkage)
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

final_spec_xgb



#Giving the specific to our Test data set:

set.seed(877)

final_fit_xgb <- last_fit(final_spec_xgb,
                corn_recipe_xgb,
                split = corn_split_xgb)    #The "split" will grab out test data set

final_fit_xgb %>%
  collect_predictions()


test_met_xgb <- final_fit_xgb %>%
  collect_metrics()

test_met_xgb



final_spec_xgb %>%
  fit(yield_mg_ha ~ .,
      data = bake(corn_prep_xgb, 
                  corn_train_xgb)) %>%
  augment(new_data = bake(corn_prep_xgb, 
                          corn_train_xgb)) %>% 
  rmse(yield_mg_ha, .pred) %>%
  bind_rows(
    
    
# R2
final_spec_xgb %>%
  fit(yield_mg_ha ~ .,
      data = bake(corn_prep_xgb, 
                  corn_train_xgb)) %>%
  augment(new_data = bake(corn_prep_xgb, 
                          corn_train_xgb)) %>% 
  rsq(yield_mg_ha, .pred))



publication_ready_xgb <- final_fit_xgb %>%
  collect_predictions() %>%
  ggplot(aes(x = yield_mg_ha, 
             y = .pred)) +
  geom_point(aes(fill = yield_mg_ha),
             alpha = 0.5,
             shape = 25,
             show.legend = F) +
  scale_fill_viridis_c(option = "H") +
  geom_abline(slope = 1, 
              intercept = 0, 
              color = "red", 
              linetype = 2) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Observed Yield (Mg/ha)",
    y = "Predicted Yield (Mg/ha)"
  ) +
  annotate("label", 
           x = Inf,
           y = -Inf,
           label = paste0("R-sq: ",
                          round(test_met_xgb$.estimate[[2]], 2),
                          "\nRMSE: ",
                          round(test_met_xgb$.estimate[[1]], 2)),
           hjust = 1, vjust = -0.5) +
  theme(panel.background = element_rect(fill = "gray80"),
        panel.grid = element_blank())

cat(paste0("Saving...... publication ready plot for test dataset\n"))

ggsave(plot = publication_ready_xgb,
       path = here("output", "png"),
       filename = "model_perf_test_data_xgb.png",
       height = 6,
       width = 9,
       dpi = 600)


vip_xgb <- final_spec_xgb %>%
  fit(yield_mg_ha ~ .,
    data = bake(corn_prep_xgb, corn_train_xgb)) %>% #There little change in variable importance if you use full dataset
    vi() %>%
    mutate(
      Variable = fct_reorder(Variable, 
                           Importance)) %>%
  ggplot(aes(x = Importance, 
             y = Variable),
         fill = Importance) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = NULL) +
  theme_minimal()

cat(paste0("Saving...... publication ready plot for variable importance\n"))

ggsave(plot = vip_xgb,
       path = here("output", "png"),
       filename = "vip_test_data_xgb.png",
       height = 6,
       width = 9,
       dpi = 600)


library(tidymodels)   # Core framework for modeling (includes recipes, workflows, parsnip, etc.)
library(finetune)     # Additional tuning strategies (e.g., racing, ANOVA-based tuning)
library(vip)          # For plotting variable importance from fitted models
library(ranger)       # Fast implementation of Random Forests
library(tidyverse)    # Data wrangling and visualization
library(doParallel)   # For parallel computing (useful during resampling/tuning)
library(caret)       # Other great library for Machine Learning 
library(kernlab)      # SVM engine used by tidymodels
library(here)



set.seed(9678965) # Setting seed to get reproducible results 

corn_split_svm <- initial_split(corn_training,
                            prop = .7,                 # proportion of split same as previous codes
                            strata = yield_mg_ha) 
  
corn_split_svm


corn_train_svm <- training(corn_split_svm)  # 70% of data

corn_train_svm 

corn_test_svm <- testing(corn_split_svm)

corn_test_svm



 density_plot_svm <- ggplot() +
  geom_density(data = corn_train_svm, 
               aes(x = yield_mg_ha),
               color = "red") +
  geom_density(data = corn_test_svm, 
               aes(x = yield_mg_ha),
               color = "blue") 

cat(paste0("Saving...... density plot of train and test data"))

ggsave(plot = density_plot_svm,
       path = here("output", "png"),
       filename = "density_plot_test_train_svm.png",
       height = 6,
       width = 9,
       dpi = 600)



# Create recipe for data preprocessing
corn_recipe_svm <- recipe(yield_mg_ha ~ ., data = corn_train_svm) %>% 
  step_impute_median(all_numeric_predictors()) %>%   #changing NAs values to median values
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

corn_recipe_svm


# Prep the recipe to estimate any required statistics
corn_prep_svm <- corn_recipe_svm %>% 
  prep()

# Examine preprocessing steps
corn_prep_svm



svm_spec <- # Specifying SVM as our model type, asking to tune the hyperparameters
  svm_rbf(                 # Most common choice for nonlinear regression
    cost = tune(),        # Controls model complexity
    rbf_sigma = tune()    # Controls the smoothness of the radial kernel
  ) %>%
  # specify engine
  set_engine("kernlab") %>%
  # Set to mode
  set_mode("regression")

svm_spec



set.seed(657) 

resampling_foldcv_svm <- vfold_cv(corn_train_svm, # Create 10-fold cross-validation resampling object from training data
                              v = 5)



svm_grid <- grid_space_filling(
  cost(),
  rbf_sigma(),
  size = 30
)


#Detecting cores in Sapelo:

n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))

if (is.na(n_cores)) n_cores <- parallel::detectCores() -1

#Start the cluster
cl <- makePSOCKcluster(n_cores)

registerDoParallel(cl)

cat(paste0("\nFound and registered", n_cores, "cores to work with\n"))


set.seed(6576)

# Creating the list of cross validation techniques so we can loop through them
cv_list_svm <- list(
  vfold = resampling_foldcv_svm)

#Create a empty list to store the results from the loop
results_svm <- list()

#Create the loop
for (i in seq_along(cv_list_svm)) {
  
  name_svm <- names(cv_list_svm)[i]
  
  results_svm[[name_svm]] <- tune_race_anova(object = svm_spec,
                                     preprocessor = corn_recipe_svm,
                                     resamples = cv_list_svm[[i]],
                                     grid = svm_grid,
                                     control = control_race(save_pred = TRUE,
                                                            parallel_over = "everything"))
}

stopCluster(cl)



#Create a data frame structure from the list obtained after running the loop

results_df_svm <- tibble(method = names(results_svm),
                         diff_cv = results_svm) %>%

#Collecting the metrices for each cross validation techniques using map function

mutate(metrices = map2(diff_cv, method,
                       ~.x %>%
                         collect_metrics() %>%
                         mutate(method = .y, .before = "cost")))

results_df_svm$metrices[[1]]

results_df_svm

#Bind all the metrices together to select the best performing one

all_metrices_svm <- do.call(bind_rows,
                            results_df_svm$metrices)


#Automating to pull the best method out of 3 we ran

best_method_svm <- all_metrices_svm %>%
  filter(.metric =="rmse") %>%
  slice_min(mean, n = 1) %>%
  pull(method)

#Getting the metrice (hyperparameter values of the best performing cross validation)

best_cv_object_svm <- results_df_svm %>%
  filter(method == best_method_svm) %>%
  pull(diff_cv) %>%
  first()




# Based on lowest RMSE
best_rmse_svm <- best_cv_object_svm %>% 
  select_best(metric = "rmse") %>% 
  mutate(source = "best_rmse")

best_rmse_svm



# Based on greatest R2
best_r2_svm <- best_cv_object_svm %>% 
  select_best(metric = "rsq") %>% 
  mutate(source = "best_r2")

best_r2_svm



best_rmse_svm %>% 
  bind_rows(best_rmse_svm, 
            best_r2_svm) %>%
  dplyr::select(source, everything())



final_spec_svm <- svm_rbf(
  cost = best_rmse_svm$cost,               # Cost parameter
  rbf_sigma = best_rmse_svm$rbf_sigma      # Radial basis kernel parameter
) %>%
  set_engine("kernlab") %>%
  set_mode("regression")

final_spec_svm



#Giving the specific to our Test data set:

set.seed(877)

  final_fit_svm <- last_fit(final_spec_svm,
                            corn_recipe_svm,
                            split = corn_split_svm)     #The "split" will grab out test data set

final_fit_svm %>%
  collect_predictions()



   test_met_svm <- final_fit_svm %>%
     collect_metrics()

 test_met_svm

   
final_spec_svm %>%
  fit(yield_mg_ha ~ .,
      data = bake(corn_prep_svm, 
                  corn_train_svm)) %>%
  augment(new_data = bake(corn_prep_svm, 
                          corn_train_svm)) %>% 
  rmse(yield_mg_ha, .pred) %>%
  bind_rows(
    
    # R2
    final_spec_svm %>%
      fit(yield_mg_ha ~ .,
          data = bake(corn_prep_svm, 
                      corn_train_svm)) %>%
      augment(new_data = bake(corn_prep_svm, 
                              corn_train_svm)) %>% 
      rsq(yield_mg_ha, .pred))



publication_ready_svm <- final_fit_svm %>%
  collect_predictions() %>%
  ggplot(aes(x = yield_mg_ha, 
             y = .pred)) +
  geom_point(aes(fill = yield_mg_ha),
             alpha = 0.5,
             shape = 25,
             show.legend = F) +
  scale_fill_viridis_c(option = "H") +
  geom_abline(slope = 1, 
              intercept = 0, 
              color = "red", 
              linetype = 2) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Observed Yield (Mg/ha)",
    y = "Predicted Yield (Mg/ha)"
  ) +
  annotate("label", 
           x = Inf,
           y = -Inf,
           label = paste0("R-sq: ",
                          round(test_met_svm$.estimate[[2]], 2),
                          "\nRMSE: ",
                          round(test_met_svm$.estimate[[1]], 2)),
           hjust = 1, vjust = -0.5) +
  theme(panel.background = element_rect(fill = "gray80"),
        panel.grid = element_blank())

cat(paste0("Saving...... publication ready plot for test dataset\n"))

ggsave(plot = publication_ready_svm,
       path = here("output", "png"),
       filename = "model_perf_test_data_svm.png",
       height = 6,
       width = 9,
       dpi = 600)


svm_fit <- final_spec_svm %>%
  fit(
    yield_mg_ha ~ .,
    data = bake(corn_prep_svm, corn_train_svm)
  )

train_baked_svm <- bake(corn_prep_svm, corn_train_svm)

vip::vi_permute(
  object = svm_fit,
  train = train_baked_svm,
  target = "yield_mg_ha",
  metric = "rmse",
  pred_wrapper = function(object, newdata) {
    predict(object, newdata)$.pred
  },
  nsim = 10)



final_wf_xgb <- workflow() %>%
  add_recipe(corn_recipe_xgb) %>%
  add_model(final_spec_xgb)

final_wf_svm <- workflow() %>%
  add_recipe(corn_recipe_svm) %>%
  add_model(final_spec_svm)



#Training the final model on all available training data, not only the 70% split

final_model_xgb <- final_wf_xgb %>%
  fit(data = corn_training)

final_model_svm <- final_wf_svm %>%
  fit(data = corn_training)


#Loading new Test dataset:

 corn_test_2024 <- read.csv(here("data", "corn_test_2024.csv"))



#Predicting with XGBoost:

pred_xgb <- predict(final_model_xgb, corn_test_2024) %>%
  rename(pred_xgb = .pred)

#Predicting with SVM:

pred_svm <- predict(final_model_svm, corn_test_2024) %>%
  rename(pred_svm = .pred)

#Combining predictions with the original data

new_data_predictions <- corn_test_2024 %>%
  bind_cols(pred_xgb, pred_svm)

new_data_predictions

write.csv(new_data_predictions,here("output", "new_data_predictions_xgb_svm.csv"),row.names = FALSE)



#From ".qmd" to ".R":

#knitr::purl(here("code", "XGB and SVM Machine Models.qmd"), output = here("code", "XGB_SVM_Sapelo_script.R"), documentation = 0)


