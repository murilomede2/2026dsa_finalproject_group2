# app.R

library(shiny)
library(tidyverse)
library(ggridges)
library(plotly)

#---------------------------
# Load data
#---------------------------
corn_data <- read_csv("../2026_dsa-main/2026_dsa/06_finalproject/2026dsa_finalproject_group2/data/corn_training.csv",
                      show_col_types = FALSE)

#---------------------------
# Variables
#---------------------------
soil_vars <- c("soil_ph", "om_pct", "soilk_ppm", "soilp_ppm")

weather_vars <- names(corn_data) %>%
  str_subset("^(prcp_mm|tmax_deg_c_mean|tmin_deg_c_mean|srad_w_m_2_mean|gdd)_")

month_levels <- c("Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

weather_long <- corn_data %>%
  select(all_of(weather_vars)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable_month",
    values_to = "value"
  ) %>%
  mutate(
    month = str_extract(variable_month, "[A-Z][a-z]{2}$"),
    variable = str_remove(variable_month, "_[A-Z][a-z]{2}$"),
    month = factor(month, levels = rev(month_levels)),
    variable = recode(
      variable,
      "prcp_mm" = "Precipitation",
      "tmax_deg_c_mean" = "Max Temp",
      "tmin_deg_c_mean" = "Min Temp",
      "srad_w_m_2_mean" = "Solar Radiation",
      "gdd" = "GDD"
    )
  ) %>%
  drop_na(value)

soil_long <- corn_data %>%
  select(all_of(soil_vars)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "soil_variable",
    values_to = "value"
  ) %>%
  drop_na(value) %>%
  mutate(
    soil_variable = recode(
      soil_variable,
      "soil_ph" = "Soil pH",
      "om_pct" = "Organic Matter (%)",
      "soilk_ppm" = "Soil K (ppm)",
      "soilp_ppm" = "Soil P (ppm)"
    )
  )

#---------------------------
# UI
#---------------------------
ui <- fluidPage(
  
  titlePanel("Corn Yield Exploratory Data Analysis"),
  
  sidebarLayout(
    
    sidebarPanel(
      
      selectInput(
        inputId = "eda_var",
        label = "Select variable to compare with yield:",
        choices = names(corn_data)[names(corn_data) != "yield_mg_ha"],
        selected = "soil_ph"
      ),
      
      selectInput(
        inputId = "weather_type",
        label = "Select weather variable for density plot:",
        choices = unique(weather_long$variable),
        selected = "Precipitation"
      ),
      
      selectInput(
        inputId = "soil_type",
        label = "Select soil variable for density plot:",
        choices = unique(soil_long$soil_variable),
        selected = "Soil pH"
      )
    ),
    
    mainPanel(
      tabsetPanel(
        
        tabPanel(
          "Yield Distribution",
          plotOutput("yield_hist", height = "450px"),
          plotOutput("yield_by_site", height = "500px")
        ),
        
        tabPanel(
          "Interactive Yield Plot",
          plotlyOutput("interactive_yield", height = "600px")
        ),
        
        tabPanel(
          "Weather Density Plot",
          plotOutput("weather_ridge", height = "650px")
        ),
        
        tabPanel(
          "Soil Density Plot",
          plotOutput("soil_density", height = "500px")
        )
      )
    )
  )
)

#---------------------------
# Server
#---------------------------
server <- function(input, output, session) {
  
  output$yield_hist <- renderPlot({
    ggplot(corn_data, aes(x = yield_mg_ha)) +
      geom_histogram(bins = 40, fill = "darkgreen", color = "white") +
      labs(
        title = "Distribution of Corn Yield",
        x = "Yield (Mg/ha)",
        y = "Number of Observations"
      ) +
      theme_minimal()
  })
  
  output$yield_by_site <- renderPlot({
    ggplot(corn_data, aes(x = site, y = yield_mg_ha)) +
      geom_boxplot(fill = "lightblue") +
      labs(
        title = "Corn Yield Distribution by Site",
        x = "Site",
        y = "Yield (Mg/ha)"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  output$interactive_yield <- renderPlotly({
    
    p <- ggplot(corn_data, aes(x = .data[[input$eda_var]], y = yield_mg_ha)) +
      geom_point(alpha = 0.4, color = "darkblue") +
      labs(
        title = paste("Yield vs", input$eda_var),
        x = input$eda_var,
        y = "Yield (Mg/ha)"
      ) +
      theme_minimal()
    
    ggplotly(p)
  })
  
  output$weather_ridge <- renderPlot({
    
    weather_long %>%
      filter(variable == input$weather_type) %>%
      ggplot(aes(x = value, y = month, fill = month)) +
      geom_density_ridges(alpha = 0.75, scale = 1.2) +
      labs(
        title = paste("Density Distribution of", input$weather_type, "by Month"),
        x = input$weather_type,
        y = "Month"
      ) +
      theme_minimal() +
      theme(legend.position = "none")
  })
  
  output$soil_density <- renderPlot({
    
    soil_long %>%
      filter(soil_variable == input$soil_type) %>%
      ggplot(aes(x = value)) +
      geom_density(fill = "darkorange", alpha = 0.5, color = "black") +
      labs(
        title = paste("Density Distribution of", input$soil_type),
        x = input$soil_type,
        y = "Density"
      ) +
      theme_minimal()
  })
}

#---------------------------
# Run app
#---------------------------
shinyApp(ui = ui, server = server)