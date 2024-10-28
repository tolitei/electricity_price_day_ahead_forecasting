# EDA - Electricity price (day-ahead forecast)

## OVERVIEW
The article ["Tackling Climate Change with Machine Learning"](https://dl.acm.org/doi/10.1145/3485128). highlights how AI can significantly reduce greenhouse gas emissions by improving electricity systems. In a nutshell:
- Electricity generation is a major source of greenhouse gas emissions.
- Reducing emissions requires a shift to low-carbon energy sources and increased system flexibility to meet demand without relying on fossil-fuel backup.

*The Importance of Day-Ahead Forecasting*: According to [OMIE](https://www.omie.es/en/mercado-de-electricidad):
-  "Every day at 12:00 CET, the day-ahead market session occurs, where electricity prices and energy allocations are determined for the next 24 hours across Europe".
- Accurate forecasting during this session is essential for effective energy management and reducing emissions.

### ML Role
- Improved Forecasting: ML can better predict energy needs, allowing system operators to reduce reliance on polluting backup plants.
- Accurate ML forecasts guide where and when to build renewable plants for maximum efficiency.

## Table of Contents
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Libraries used](#libraries-used)
- [Benefits of Doing This Project](#benefits-of-doing-this-project)
- [Repos/Notebooks that inspired me](#other-reposnotebooks-that-inspired-me)


## Dataset
- The Dataset is available on [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather). 
- The data used for this project contains 4 years of electrical consumption, generation and pricing data for Spain.
- Consumption and generation data retrieved from ENTSOE (European Network of Transmission System Operators).

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **General Information**: Gather and summarize the key characteristics of the dataset, including the number of observations, features, and data types.
- **Missing Data Identification**;
- **Intermittency Check**: Assess the data for sparsity, which could affect the reliability of forecasting models;
- **Outlier Detection**;
- **Visual Exploration**: Create visualizations (e.g., histograms, scatter plots, and time series plots) to uncover patterns, trends, and relationships within the data;
- **Conclusion**: Summarize the findings from the EDA, highlighting key insights and implications for the modeling phase.

## Libraries used

**1. Exploratory Data Analysis (EDA)**:

- pandas: For data manipulation and analysis.
- missingno: To visualize and analyze missing data patterns.
- statsmodels: For statistical modeling and hypothesis testing.
- seaborn, matplotlib: For making visualizations graphics with ease.
- ydata-profiling: A powerful low-code solution used at the end of the analysis process. Excellent tool for quick exploratory analysis and understanding of dataset characteristics.
     
**2. Documentation Inspiration**: 
- Creating README files: [Best Practices for Writing a README](https://www.dhiwise.com/post/how-to-write-a-readme-that-stands-out-in-best-practices) | [Basic writing and formatting syntax](https://docs.github.com/pt/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

## Benefits of Doing This Project

- **Hands-On Learning**: Engaging in this project provides a practical learning experience, allowing you to apply theoretical concepts to real-world data and scenarios.

- **Research and Insights**: By exploring other repositories, you’ll gain insights into prevalent techniques and methodologies used in day-ahead electricity price forecasting, enhancing your understanding of best practices in the field.

- **Application of Proven Techniques**: You will have the opportunity to implement some of the most widely-used forecasting methods, solidifying your knowledge and skills in predictive analytics.

- **Deeper Understanding of the Electricity Market**: This project will deepen your understanding of how the electricity market operates, including factors influencing price fluctuations and the dynamics of supply and demand.

## Other repos/notebooks that inspired me
- Electricity price forecasting with DNNs [Kaggle kernel of Dimitrios Roussis](https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda/notebook#1.1.-Energy-dataset).
- [Forecasting Day-Ahead Electricity Prices in Spain](https://github.com/magnusbig/forecasting_day_ahead_electric_prices_in_spain) by @magnusbig. Special thanks to [@magnusbig](https://github.com/magnusbig) for the inspiration provided by this repository!

Take a moment to check out their work and appreciate the contributions they’ve made.

## Next Steps

**Machine Learning Modeling**: 
   - Develop baseline models to establish a performance benchmark.
   - Introduce feature engineering techniques commonly used in day-ahead forecasting to enhance model accuracy and performance.
   - Utilize the methodology inspired by Jean-François Puget, PhD (Kaggle Grandmaster), as outlined in this informative video: [Machine Learning Modeling Techniques](https://www.youtube.com/watch?v=vJMoE3FMZXM). This resource provides valuable insights and practical approaches for effective model building.

**Model Evaluation**: 
   - Implement evaluation metrics to assess model performance and robustness.
   
**Feature Importance Analysis**: 
   - Explore and apply techniques to determine the significance of each feature in the model.
   - Utilize insights from feature importance to refine the feature set, potentially improving future models and forecasting accuracy.

**Incorporate Numerical Weather Prediction (NWP) Data**
   - Integrate NWP data to evaluate its impact on forecasting accuracy and model performance.
