# app.R

library(shiny)
library(readr)
library(dplyr)
library(ggplot2)
library(ggridges)

#---------------------------
# File paths
#---------------------------
data_path <- "data/corn_training.csv"

xgb_perf_png <- "data/model_perf_test_data_xgb_300dpi.png"
xgb_vip_png  <- "data/vip_test_data_xgb_300dpi.png"

#---------------------------
# Load only needed columns
#---------------------------
all_names <- names(read_csv(
  data_path,
  n_max = 0,
  show_col_types = FALSE
))

weather_cols <- grep(
  "^(prcp_mm|tmax_deg_c_mean|tmin_deg_c_mean|srad_w_m_2_mean|gdd)_",
  all_names,
  value = TRUE
)

soil_cols <- c("soil_ph", "om_pct", "soilk_ppm", "soilp_ppm")
soil_cols <- soil_cols[soil_cols %in% all_names]

needed_cols <- c("site", "yield_mg_ha", weather_cols, soil_cols)

corn_data <- read_csv(
  data_path,
  col_select = all_of(needed_cols),
  show_col_types = FALSE
)

corn_data$site <- as.factor(corn_data$site)

#---------------------------
# Labels
#---------------------------
month_levels <- c("Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

weather_labels <- c(
  "prcp_mm" = "Precipitation",
  "tmax_deg_c_mean" = "Max Temp",
  "tmin_deg_c_mean" = "Min Temp",
  "srad_w_m_2_mean" = "Solar Radiation",
  "gdd" = "GDD"
)

soil_labels <- c(
  "soil_ph" = "Soil pH",
  "om_pct" = "Organic Matter (%)",
  "soilk_ppm" = "Soil K (ppm)",
  "soilp_ppm" = "Soil P (ppm)"
)

soil_labels <- soil_labels[names(soil_labels) %in% soil_cols]

soil_colors <- c(
  "soil_ph" = "darkorange",
  "om_pct" = "forestgreen",
  "soilk_ppm" = "steelblue",
  "soilp_ppm" = "purple"
)

#---------------------------
# Weather choices
#---------------------------
weather_prefix <- unique(sub("_[A-Z][a-z]{2}$", "", weather_cols))

weather_choices <- setNames(
  weather_prefix,
  weather_labels[weather_prefix]
)

weather_choices <- weather_choices[!is.na(names(weather_choices))]

#---------------------------
# UI
#---------------------------
ui <- fluidPage(
  
  titlePanel("Corn Yield Exploratory Data Analysis"),
  
  sidebarLayout(
    
    sidebarPanel(
      selectInput(
        inputId = "weather_type",
        label = "Select weather variable:",
        choices = weather_choices,
        selected = "prcp_mm"
      ),
      
      hr(),
      
      selectInput(
        inputId = "soil_type",
        label = "Select soil variable:",
        choices = soil_labels,
        selected = soil_labels[1]
      )
    ),
    
    mainPanel(
      tabsetPanel(
        
        tabPanel(
          "Yield Distribution",
          plotOutput("yield_by_site", height = "500px")
        ),
        
        tabPanel(
          "Weather Density Plot",
          plotOutput("weather_density", height = "650px")
        ),
        
        tabPanel(
          "Soil Density Plot",
          plotOutput("soil_density", height = "550px")
        ),
        
        tabPanel(
          "Predicted vs Observed Yield - XGBoost",
          h3("Predicted vs Observed Yield - XGBoost"),
          uiOutput("xgb_perf_message"),
          imageOutput("xgb_perf_plot", height = "700px")
        ),
        
        tabPanel(
          "Variable Importance - XGBoost",
          h3("Variable Importance - XGBoost"),
          uiOutput("xgb_vip_message"),
          imageOutput("xgb_vip_plot", height = "700px")
        )
      )
    )
  )
)

#---------------------------
# Server
#---------------------------
server <- function(input, output, session) {
  
  output$yield_by_site <- renderPlot({
    ggplot(corn_data, aes(x = site, y = yield_mg_ha)) +
      geom_boxplot(fill = "lightblue", outlier.size = 0.2) +
      labs(
        title = "Corn Yield Distribution by Site",
        x = "Site",
        y = "Yield (Mg/ha)"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  output$weather_density <- renderPlot({
    
    selected_weather_prefix <- input$weather_type
    selected_weather_label <- weather_labels[[selected_weather_prefix]]
    
    selected_cols <- weather_cols[
      sub("_[A-Z][a-z]{2}$", "", weather_cols) == selected_weather_prefix
    ]
    
    weather_plot_data <- data.frame()
    
    for (col in selected_cols) {
      
      month_name <- sub(".*_([A-Z][a-z]{2})$", "\\1", col)
      
      temp <- data.frame(
        value = corn_data[[col]],
        month = factor(month_name, levels = rev(month_levels))
      )
      
      weather_plot_data <- rbind(weather_plot_data, temp)
    }
    
    weather_plot_data <- weather_plot_data[!is.na(weather_plot_data$value), ]
    
    ggplot(weather_plot_data, aes(x = value, y = month, fill = month)) +
      geom_density_ridges(
        alpha = 0.75,
        scale = 1.2,
        color = "black",
        linewidth = 0.3
      ) +
      labs(
        title = paste("Density Distribution of", selected_weather_label, "by Month"),
        x = selected_weather_label,
        y = "Month"
      ) +
      theme_minimal() +
      theme(
        legend.position = "none",
        plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10)
      )
  })
  
  output$soil_density <- renderPlot({
    
    selected_soil_var <- names(soil_labels)[soil_labels == input$soil_type]
    
    ggplot(corn_data, aes(x = .data[[selected_soil_var]])) +
      geom_density(
        fill = soil_colors[[selected_soil_var]],
        color = soil_colors[[selected_soil_var]],
        alpha = 0.65,
        linewidth = 1,
        na.rm = TRUE
      ) +
      labs(
        title = paste("Density Distribution of", input$soil_type),
        x = input$soil_type,
        y = "Density"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10)
      )
  })
  
  output$xgb_perf_message <- renderUI({
    if (!file.exists(xgb_perf_png)) {
      tags$p(
        style = "color:red;",
        paste("Image not found at:", normalizePath(xgb_perf_png, mustWork = FALSE))
      )
    }
  })
  
  output$xgb_perf_plot <- renderImage({
    req(file.exists(xgb_perf_png))
    
    list(
      src = normalizePath(xgb_perf_png, mustWork = TRUE),
      contentType = "image/png",
      width = 900,
      alt = "Predicted vs Observed Yield - XGBoost"
    )
  }, deleteFile = FALSE)
  
  output$xgb_vip_message <- renderUI({
    if (!file.exists(xgb_vip_png)) {
      tags$p(
        style = "color:red;",
        paste("Image not found at:", normalizePath(xgb_vip_png, mustWork = FALSE))
      )
    }
  })
  
  output$xgb_vip_plot <- renderImage({
    req(file.exists(xgb_vip_png))
    
    list(
      src = normalizePath(xgb_vip_png, mustWork = TRUE),
      contentType = "image/png",
      width = 900,
      alt = "Variable Importance - XGBoost"
    )
  }, deleteFile = FALSE)
}

shinyApp(ui = ui, server = server)