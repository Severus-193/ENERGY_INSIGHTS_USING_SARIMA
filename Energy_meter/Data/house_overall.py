import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import os
import warnings
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

warnings.filterwarnings('ignore')

# Define the function to load datasets for a given house number
def load_data_for_house(house_number):
    daily_data = pd.read_csv(f'daily_data_house{house_number}.csv')
    daily_data['time'] = pd.to_datetime(daily_data['time'])
    daily_data.set_index('time', inplace=True)
    daily_data = daily_data.asfreq('D')

    hourly_data = pd.read_csv(f'hourly_data_house{house_number}.csv')
    hourly_data['time'] = pd.to_datetime(hourly_data['time'])
    hourly_data.set_index('time', inplace=True)
    hourly_data = hourly_data.asfreq('H')

    return daily_data, hourly_data

# List of appliances and house overall
appliances = ["Air_Conditioner [kW]","Fans [kW]","Tubelight [kW]","Television [kW]","Washing_Machine [kW]","Water_Geyser [kW]","Mixer_Grinder [kW]","Wet_Grinder [kW]","Refrigerator [kW]","Microwave_Oven [kW]","Dishwasher [kW]"]
house_overall = "House overall [kW]"

# Function to fit SARIMA model and make predictions
def fit_sarima(data, appliance, p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_period, prediction_days):
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    # Grid search for best hyperparameters
    for p, d, q, sp, sd, sq in product(p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values):
        try:
            model = sm.tsa.statespace.SARIMAX(data[appliance], order=(p, d, q), seasonal_order=(sp, sd, sq, seasonal_period))
            model_fit = model.fit(disp=False)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p, d, q)
                best_seasonal_order = (sp, sd, sq, seasonal_period)
                best_model = model_fit
        except:
            continue

    if best_model is not None:
        predictions = best_model.predict(len(data), len(data) + prediction_days - 1)
        return predictions, best_model
    else:
        return None, None

# Define hyperparameter ranges (reduced for speed)
p_values = range(0, 2)
d_values = range(0, 2)
q_values = range(0, 2)
seasonal_p_values = range(0, 2)
seasonal_d_values = range(0, 2)
seasonal_q_values = range(0, 2)
seasonal_period = 7  # Weekly seasonality

# Function to predict for each appliance for a given house
def predict_for_house(house_number):
    daily_data, hourly_data = load_data_for_house(house_number)
    predictions = {}
    models = {}

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda appliance: (appliance, fit_sarima(daily_data, appliance, p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_period, prediction_days=60)), appliances))
        for appliance, (pred, model) in results:
            if pred is not None:
                predictions[appliance] = pred
                models[appliance] = model
            else:
                print(f"No suitable model found for {appliance} in house {house_number}")

    # Predict using house overall for the next 60 days (two months)
    predictions_house, model_house = fit_sarima(daily_data, house_overall, p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_period, prediction_days=60)
    
    return daily_data, hourly_data, predictions, predictions_house, models, model_house


# Perform predictions only for houses with available datasets
houses_data = {}
for house_number in range(1, 1001):  # Range increased to 1000
    daily_data_path = f'daily_data_house{house_number}.csv'
    hourly_data_path = f'hourly_data_house{house_number}.csv'
    
    # Check if both daily and hourly data files exist for the house
    if os.path.exists(daily_data_path) and os.path.exists(hourly_data_path):
        houses_data[house_number] = predict_for_house(house_number)
    else:
        print(f"Data not available for house {house_number}")


# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
# App layout
app.layout = html.Div(
    [
        html.H1("Energy Consumption and Cost Prediction"),
        html.Div(id='output-statements'),
        dcc.Graph(id='house-overall-graph'),
        html.H2("Appliance Predictions"),
        html.Div(
            [dcc.Graph(id=f'graph-{i}') for i in range(len(appliances))],
            style={'textAlign': 'center'}
        )
    ],
    style={
        'backgroundColor': 'skyblue',
        'fontFamily': 'Times New Roman',
        'textAlign': 'center'
    }
)


# Define the billing function according to the given rates
def calculate_bill(kwh):
    if kwh <= 100:
        return 0
    elif kwh <= 400:
        return (kwh - 100) * 4.70
    elif kwh <= 500:
        return (400 - 100) * 4.70 + (kwh - 400) * 6.30
    elif kwh <= 600:
        return (400 - 100) * 4.70 + (500 - 400) * 6.30 + (kwh - 500) * 8.40
    elif kwh <= 800:
        return (400 - 100) * 4.70 + (500 - 400) * 6.30 + (600 - 500) * 8.40 + (kwh - 600) * 9.45
    elif kwh <= 1000:
        return (400 - 100) * 4.70 + (500 - 400) * 6.30 + (600 - 500) * 8.40 + (800 - 600) * 9.45 + (kwh - 800) * 10.50
    else:
        return (400 - 100) * 4.70 + (500 - 400) * 6.30 + (600 - 500) * 8.40 + (800 - 600) * 9.45 + (1000 - 800) * 10.50 + (kwh - 1000) * 11.55

@app.callback(
    Output('output-statements', 'children'),
    [Input('house-overall-graph', 'id')]
)
def update_output(_):
    output_statements = []
    for house_number, (daily_data, hourly_data, predictions, predictions_house, models, model_house) in houses_data.items():
        total_predicted_consumption_house_kwh = predictions_house.sum()
        total_cost_house_inr = calculate_bill(total_predicted_consumption_house_kwh)
        total_power_consumption_kwh = sum(predictions[appliance].sum() for appliance in appliances if appliance in predictions)
        
        results = {}
        for appliance in appliances:
            if appliance in predictions:
                total_predicted_consumption_kwh = predictions[appliance].sum()
                results[appliance] = {
                    "total_kwh": total_predicted_consumption_kwh,
                    "percentage": (total_predicted_consumption_kwh / total_power_consumption_kwh) * 100
                }

        peak_hours = {}
        off_peak_hours = {}

        for appliance in appliances:
            if appliance in hourly_data:
                usage = hourly_data[appliance].groupby(hourly_data.index.hour).mean()
                peak_hour = usage.idxmax()
                off_peak_hour = usage.idxmin()
                peak_hours[appliance] = peak_hour
                off_peak_hours[appliance] = off_peak_hour

        house_output_statements = [
            f"House {house_number}:",
            f"1. Total power consumption for the next 60 days (two months) for each appliance:"
        ]
        for appliance in appliances:
            if appliance in results:
                house_output_statements.append(f"   - {appliance.replace(' [kW]', '')}: {results[appliance]['total_kwh']:.2f} kWh ({results[appliance]['percentage']:.2f}%)")
        house_output_statements.append(f"   - House overall: {total_predicted_consumption_house_kwh:.2f} kWh")
        house_output_statements.append(f"2. Your electricity bill for the next two months using House overall is estimated to be ₹{total_cost_house_inr:.2f} if current usage patterns continue.")

        output_statements.append(html.Div([html.P(statement) for statement in house_output_statements]))

    return output_statements

# Update graphs
@app.callback(
    [Output('house-overall-graph', 'figure')] + [Output(f'graph-{i}', 'figure') for i in range(len(appliances))],
    [Input('house-overall-graph', 'id')]
)
def update_graphs(_):
    figures = []
    
    # Plot for house overall
    fig_house = go.Figure()
    for house_number, (daily_data, _, predictions, predictions_house, _, _) in houses_data.items():
        fig_house.add_trace(go.Scatter(x=daily_data.index, y=daily_data[house_overall], mode='lines', name=f'House {house_number} {house_overall} Training Data'))
        fig_house.add_trace(go.Scatter(x=predictions_house.index, y=predictions_house, mode='lines', name=f'House {house_number} {house_overall} Predictions'))
    fig_house.update_layout(
        title=f"Training Data vs Predictions for {house_overall.replace(' [kW]', '')}",
        xaxis_title="Date",
        yaxis_title="Power [kW]",
        legend=dict(x=0, y=1, traceorder="normal"),
        width=1200,
        height=800
    )
    figures.append(fig_house)
    
    # Plots for appliances
    for i, appliance in enumerate(appliances):
        fig = go.Figure()
        for house_number, (daily_data, _, predictions, _, _, _) in houses_data.items():
            if appliance in predictions:
                fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data[appliance], mode='lines', name=f'House {house_number} {appliance} Training Data'))
                fig.add_trace(go.Scatter(x=predictions[appliance].index, y=predictions[appliance], mode='lines', name=f'House {house_number} {appliance} Predictions'))
        fig.update_layout(
            title=f"Training Data vs Predictions for {appliance.replace(' [kW]', '')}",
            xaxis_title="Date",
            yaxis_title="Power [kW]",
            legend=dict(x=0, y=1, traceorder="normal"),
            width=1200,
            height=800
        )
        figures.append(fig)

    return figures

if __name__ == '__main__':
    app.run_server(debug=True)