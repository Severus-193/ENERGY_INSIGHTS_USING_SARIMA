# ENERGY_INSIGHTS_USING_SARIMA


## Project Overview
This project consists of three interactive web applications developed using Dash, integrated into a website along with an Electricity Bill calculator. The project aims to analyze household power consumption, predict electricity bills using SARIMA models, and provide insights into appliance usage patterns, including peak and off-peak hours. Additionally, a chatbot is integrated to answer user queries related to power consumption, bill breakdown, and power-saving tips.

## Features
### 1. Custom Dataset Creation (energy.py)
- Users can create customized datasets of household power consumption.
- Allows selection of individual appliances and their power usage.
- Stores data for further analysis and prediction.

### 2. Electricity Bill Prediction (house_overall.py)
- Utilizes a SARIMA model to analyze power consumption patterns.
- Predicts the electricity bill based on past usage data.
- Identifies peak and off-peak hours of appliance usage.
- Provides insights into the overall household power consumption trends.

### 3. Chatbot for Power Consumption Queries (Chat.py)
- Trained with the SARIMA model to assist users.
- Provides bill breakdown for households.
- Offers insights into power consumption of individual appliances.
- Suggests power-saving tips based on usage patterns.

### 4. EB Bill Calculator(calculator.html)
- Allows users to manually input power consumption details.
- Computes the estimated electricity bill.
- Helps users compare predicted and actual electricity costs.

## Technology Stack
- **Python** (Dash, Pandas, NumPy, Statsmodels for SARIMA model)
- **Dash** (It is utilized for building interactive web applications)
- **Machine Learning** (SARIMA model for time-series forecasting)
- **HTML/CSS** (It is used for styling and front-end customization)

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Severus-193/ENERGY_INSIGHTS_USING_SARIMA.git
   cd ENERGY_INSIGHTS_USING_SARIMA.git
   ```

2. Run each Dash application separately:
   ```bash
   python energy.py # For dataset creation
   python house_overall.py  # For bill prediction
   python Chat.py  # For chatbot and power insights
   ```
3. Open the browser and navigate to the provided localhost URL.

## Usage
- Open the dataset creation app to input appliance power consumption data.
- Use the bill prediction app to analyze and forecast electricity bills.
- Interact with the chatbot to get a detailed bill breakdown and energy-saving suggestions.
- Utilize the EB bill calculator for manual cost estimation.

## Future Enhancements
- Integration of real-time electricity consumption monitoring.
- Expansion of predictive models to include additional forecasting techniques.
- Mobile-friendly UI for better accessibility.


