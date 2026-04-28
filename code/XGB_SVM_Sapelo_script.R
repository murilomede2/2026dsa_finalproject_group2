# shared_lib_path <- "/home/mm60458/instructor_data/shared_R_libs"
                      
#class_packages <- c("tidymodels", "finetune", "vip", "xgboost", "ranger", "tidyverse", "doParallel", "caret", "here", "readr", "dplyr", "kernlab")

#install.packages(class_packages, lib = shared_lib_path)

# .libPaths(c("/home/mm60458/instructor_data/shared_R_libs", .libPaths()))


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
 
gsave(plot = density_plot_xgb,
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
                              v = 5)

resampling_foldcv_xgb

xgb_grid <- grid_space_filling(
  tree_depth(),
  min_n(),
  learn_rate(),
  trees(),
  size = 5)

xgb_grid

 grid_plot_xgb <- ggplot(data = xgb_grid,
       aes(x = tree_depth, 
           y = min_n)) +
  geom_point(aes(color = factor(learn_rate),
                 size = trees),
             alpha = .5,
             show.legend = FALSE)


gsave(plot = grid_plot_xgb,
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


set.seed(6576)

xgb_res <- tune_race_anova(object = xgb_spec,
                      preprocessor =corn_recipe_xgb,
                      resamples = resampling_foldcv_xgb,
                      grid = xgb_grid,
                      control = control_race(save_pred = TRUE))

stopCluster(cl)

xgb_res



plot_race_xgb <- plot_race(xgb_res)

plot_race_xgb

gsave(
  plot = plot_race_xgb,
  path = here("output", "png"),
  filename = "race_plot_xgb.png",
  height = 6,
  width = 9,
  dpi = 600)


# Based on lowest RMSE
best_rmse_xgb <- xgb_res %>% 
  select_best(metric = "rmse")%>% 
  mutate(source = "best_rmse")

best_rmse_xgb


# Based on greatest R2
best_r2_xgb <- xgb_res %>% 
  select_best(metric = "rsq")%>% 
  mutate(source = "best_r2")

best_r2_xgb



  best_rmse_xgb %>% 
  bind_rows(best_r2_xgb)


final_spec_xgb <- boost_tree(
  trees = best_rmse_xgb$trees,           # Number of boosting rounds (trees)
  tree_depth = best_rmse_xgb$tree_depth, # Maximum depth of each tree
  min_n = best_rmse_xgb$min_n,           # Minimum number of samples to split a node
  learn_rate = best_rmse_xgb$learn_rate  # Learning rate (step size shrinkage)
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

final_spec_xgb



#Giving the specific to our data set:

set.seed(877)

final_fit_xgb <- last_fit(final_spec_xgb,
                corn_recipe_xgb,
                split = corn_split_xgb)    

final_fit_xgb


#Observed yield + predicted yield for the test data set:

final_fit_xgb %>%
  collect_predictions()


#Getting the final performance of the model with the best model chosen in chunk 4:
#This is the result we want in the slide:

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
             alpha = 0.7,
             shape = 21,
             show.legend = F) +
  scale_fill_viridis_c(option = "H") +
  geom_abline(color = "red", 
              linetype = 2) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Observed Yield (Mg/ha)",
    y = "Predicted Yield (Mg/ha)",
    title = "XGBoost: Observed vs Predicted Yield - Model Validation") +
  annotate("label",
           x = Inf,
           y = -Inf,
           label = paste0(
                          "R-sq: ", round(test_met_xgb$.estimate[test_met_xgb$.metric == "rsq"], 2),
                          "\nRMSE: ", round(test_met_xgb$.estimate[test_met_xgb$.metric == "rmse"], 2)),
                          hjust = 1,
                          vjust = -0.5) +
  theme(panel.background = element_rect(fill = "gray80"),
        panel.grid = element_blank())

gsave(plot = publication_ready_xgb,
       path = here("output", "png"),
       filename = "model_perf_test_data_xgb.png",
       height = 6,
       width = 9,
       dpi = 600)



vip_xgb <- final_spec_xgb %>%
  fit(yield_mg_ha ~ .,
    data = bake(corn_prep_xgb, corn_train_xgb)) %>% 
    vi() %>%
    mutate(
      Variable = fct_reorder(Variable, 
                           Importance)) %>%
  ggplot(aes(x = Importance, 
             y = Variable),
         fill = Importance) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = NULL,
       title = "Variable Importance in Yield Prediction") +
  theme_minimal()

gsave(plot = vip_xgb,
       path = here("output", "png"),
       filename = "vip_test_data_xgb.png",
       height = 6,
       width = 9,
       dpi = 600)


library(kernlab)      # SVM engine used by tidymodels



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


gsave(plot = density_plot_svm,
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
resampling_foldcv_svm



svm_grid <- grid_space_filling(
  cost(),
  rbf_sigma(),
  size = 20)

svm_grid


#Detecting cores in Sapelo:

n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))

if (is.na(n_cores)) n_cores <- parallel::detectCores() -1

#Start the cluster
cl <- makePSOCKcluster(n_cores)

registerDoParallel(cl)


set.seed(6576)

  svm_res <- tune_race_anova(object = svm_spec,
                                     preprocessor = corn_recipe_svm,
                                     resamples = resampling_foldcv_svm,
                                     grid = svm_grid,
                                     control = control_race(save_pred = TRUE,
                                                            parallel_over = "everything"))

stopCluster(cl)

svm_res



plot_race_xgb <- plot_race(svm_res)

ggsave(
  plot = plot_race_svm,
  path = here("output", "png"),
  filename = "race_plot_svm.png",
  height = 6,
  width = 9,
  dpi = 600)



# Based on lowest RMSE
best_rmse_svm <- svm_res %>% 
  select_best(metric = "rmse") %>% 
  mutate(source = "best_rmse")

best_rmse_svm



# Based on greatest R2
best_r2_svm <- svm_res %>% 
  select_best(metric = "rsq") %>% 
  mutate(source = "best_r2")

best_r2_svm



best_rmse_svm %>% 
  bind_rows(best_r2_svm)
  


final_spec_svm <- svm_rbf(
  cost = best_rmse_svm$cost,               # Cost parameter
  rbf_sigma = best_rmse_svm$rbf_sigma) %>%
  set_engine("kernlab") %>%
  set_mode("regression")

final_spec_svm



#Giving the specific to our Test data set:

set.seed(877)

  final_fit_svm <- last_fit(final_spec_svm,
                            corn_recipe_svm,
                            split = corn_split_svm)     

  final_fit_svm


final_fit_svm %>%
  collect_predictions()


#Getting the final performance of the model with the best model chosen in chunk 4:
#This is the result we want in the slide:

   test_met_svm <- final_fit_svm %>%
     collect_metrics()

 test_met_svm

   #The output from this chunk will be used to compare with the training set to check if the model is over fitting or can generalize for new data sets:

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
             alpha = 0.7,
             shape = 21,
             show.legend = F) +
  scale_fill_viridis_c(option = "H") +
  geom_abline(color = "red", 
              linetype = 2) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Observed Yield (Mg/ha)",
    y = "Predicted Yield (Mg/ha)",
    title = "SVM: Observed vs Predicted Yield - Model Validation") +
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

 ggsave(plot = publication_ready_svm,
       path = here("output", "png"),
       filename = "model_perf_test_data_svm.png",
       height = 6,
       width = 9,
       dpi = 600)


train_baked_svm <- bake(corn_prep_svm, corn_train_svm)

svm_fit <- final_spec_svm %>%
  fit(
    yield_mg_ha ~ .,
    data = train_baked_svm)

set.seed(657)

train_baked_svm_small <- train_baked_svm %>%
  slice_sample(n = 50000)

vi_svm <- vip::vi_permute(
  object = svm_fit,
  train = train_baked_svm_small,
  target = "yield_mg_ha",
  metric = "rmse",
  pred_wrapper = function(object, newdata) {
    predict(object, newdata)$.pred
  },
  nsim = 3)

vip_svm_plot <- vip::vip(vi_svm, num_features = 15) +
  labs(
    title = "Permutation Variable Importance - SVM",
    x = "Increase in RMSE",
    y = "Variables"
  ) +
  theme_minimal()


ggsave(
  plot = vip_svm_plot,
  path = here("output", "png"),
  filename = "variable_importance_svm.png",
  height = 6,
  width = 9,
  dpi = 600)



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

 knitr::purl(here("code", "XGB and SVM Machine Models.qmd"), output = here("code", "XGB_SVM_Sapelo_script.R"), documentation = 0)


