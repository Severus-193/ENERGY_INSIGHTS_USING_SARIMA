import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

warnings.filterwarnings('ignore')

# Define the list of all possible appliances
appliances = ["Air_Conditioner [kW]", "Fans [kW]", 
              "Tubelight [kW]", "Television [kW]", "Washing_Machine [kW]", 
              "Water_Geyser [kW]", "Mixer_Grinder [kW]", "Wet_Grinder [kW]", 
              "Refrigerator [kW]", "Microwave_Oven [kW]", "Dishwasher [kW]"]
house_overall = "House overall [kW]"  

# Initialize dictionaries to store data
daily_data = {}
hourly_data = {}
house_appliances = {}

# Load data for each house
for i in range(1, 1001):
    try:
        daily_data[i] = pd.read_csv(f'daily_data_house{i}.csv')
        daily_data[i]['time'] = pd.to_datetime(daily_data[i]['time'])
        daily_data[i].set_index('time', inplace=True)
        daily_data[i] = daily_data[i].asfreq('D')
        
        hourly_data[i] = pd.read_csv(f'hourly_data_house{i}.csv')
        hourly_data[i]['time'] = pd.to_datetime(hourly_data[i]['time'])
        hourly_data[i].set_index('time', inplace=True)
        hourly_data[i] = hourly_data[i].asfreq('H')
        
        # Get the list of appliances present in the current house
        house_appliances[i] = daily_data[i].columns.tolist()
    except FileNotFoundError:
        print(f"Data for House {i} not found.")

# Function to fit SARIMA model and make predictions
def fit_sarima(data, appliance, p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_period, prediction_days):
    if appliance not in data.columns:
        return None, None
    
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None
    
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

# Predict for each appliance for the next 60 days (two months) in parallel
def predict_for_appliance(house, appliance):
    if appliance not in house_appliances.get(house, []):
        return appliance, (None, None)
    return appliance, fit_sarima(daily_data[house], appliance, p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_period, prediction_days=60)

# Initialize predictions and models dictionaries
predictions = {house: {} for house in range(1, 1001)}
models = {house: {} for house in range(1, 1001)}

# Perform prediction for each house with available data
for house in range(1, 1001):
    if house in daily_data:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda appliance: predict_for_appliance(house, appliance), appliances))
            for appliance, (pred, model) in results:
                if pred is not None:
                    predictions[house][appliance] = pred
                    models[house][appliance] = model
                else:
                    print(f"No suitable model found for {appliance} in House {house}")

# Predict using house overall for the next 60 days (two months)
overall_predictions = {}
for house in range(1, 1001):
    if house in daily_data:
        overall_predictions[house], _ = fit_sarima(daily_data[house], house_overall, p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_period, prediction_days=60)

# Calculate power consumption and cost for house overall
total_predicted_consumption_house_kwh = {house: overall_predictions[house].sum() for house in range(1, 1001) if house in overall_predictions}

# Define the billing function according to the given rates
def calculate_bill(kwh):
    if kwh <= 100:
        cost = 0
        explanation = "You have free consumption for the first 100 units."
    elif kwh <= 200:
        cost = (kwh - 100) * 2.35
        explanation = "You are charged ₹2.35 per unit for units 101 to 200."
    elif kwh <= 400:
        cost = (100 * 2.35) + (kwh - 200) * 4.70
        explanation = "You are charged ₹2.35 per unit for units 101 to 200 and ₹4.70 per unit for units 201 to 400."
    elif kwh <= 500:
        cost = (100 * 2.35) + (200 * 4.70) + (kwh - 400) * 6.30
        explanation = "You are charged ₹2.35 per unit for units 101 to 200, ₹4.70 per unit for units 201 to 400, and ₹6.30 per unit for units 401 to 500."
    elif kwh <= 600:
        cost = (100 * 2.35) + (200 * 4.70) + (100 * 6.30) + (kwh - 500) * 8.40
        explanation = "You are charged ₹2.35 per unit for units 101 to 200, ₹4.70 per unit for units 201 to 400, ₹6.30 per unit for units 401 to 500, and ₹8.40 per unit for units 501 to 600."
    elif kwh <= 800:
        cost = (100 * 2.35) + (200 * 4.70) + (100 * 6.30) + (100 * 8.40) + (kwh - 600) * 9.45
        explanation = "You are charged ₹2.35 per unit for units 101 to 200, ₹4.70 per unit for units 201 to 400, ₹6.30 per unit for units 401 to 500, ₹8.40 per unit for units 501 to 600, and ₹9.45 per unit for units 601 to 800."
    elif kwh <= 1000:
        cost = (100 * 2.35) + (200 * 4.70) + (100 * 6.30) + (100 * 8.40) + (200 * 9.45) + (kwh - 800) * 10.50
        explanation = "You are charged ₹2.35 per unit for units 101 to 200, ₹4.70 per unit for units 201 to 400, ₹6.30 per unit for units 401 to 500, ₹8.40 per unit for units 501 to 600, ₹9.45 per unit for units 601 to 800, and ₹10.50 per unit for units 801 to 1000."
    else:
        cost = (100 * 2.35) + (200 * 4.70) + (100 * 6.30) + (100 * 8.40) + (200 * 9.45) + (200 * 10.50) + (kwh - 1000) * 11.55
        explanation = "You are charged ₹2.35 per unit for units 101 to 200, ₹4.70 per unit for units 201 to 400, ₹6.30 per unit for units 401 to 500, ₹8.40 per unit for units 501 to 600, ₹9.45 per unit for units 601 to 800, ₹10.50 per unit for units 801 to 1000, and ₹11.55 per unit for units exceeding 1000."
    return cost, explanation

# Calculate total cost for each house
total_cost_house = {house: calculate_bill(total_predicted_consumption_house_kwh[house])[0] for house in total_predicted_consumption_house_kwh}

# Example peak hours and off-peak hours (dummy data)
# Peak and off-peak hours analysis for power saving tips
peak_hours_house = {house: {} for house in range(1, 1001)}
off_peak_hours_house = {house: {} for house in range(1, 1001)}

for house in range(1, 1001):
    if house in hourly_data:  # Check if data for the house is available
        hourly_data[house]['hour'] = hourly_data[house].index.hour
        hourly_avg = hourly_data[house].groupby('hour').mean()

        for appliance in appliances:
            if appliance in hourly_avg:  # Check if data for the appliance is available
                avg_hourly_consumption = hourly_avg[appliance]
                peak_hours = avg_hourly_consumption.nlargest(3).index.tolist()
                off_peak_hours = avg_hourly_consumption.nsmallest(3).index.tolist()
                peak_hours_house[house][appliance] = peak_hours
                off_peak_hours_house[house][appliance] = off_peak_hours
            else:
                # Skip if appliance data is not available
                peak_hours_house[house][appliance] = []
                off_peak_hours_house[house][appliance] = []
    else:
        # Skip if house data is not available
        for appliance in appliances:
            peak_hours_house[house][appliance] = []
            off_peak_hours_house[house][appliance] = []


# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        'font-family': 'Times New Roman, Times, serif',
        'background-color': 'midnightblue',
        'padding': '50px',
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'height': '100vh',
        'margin': '0'
    },
    children=[
        html.Div(
            className='container',
            style={
                'max-width': '4000px',
                'padding': '50px',
                'background-color': 'mintcream',
                'text-align': 'center',
                'border-radius': '10px',
                'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'
            },
            children=[
                html.H1(
                    children='WattWatcher Plus',
                    style={'margin-bottom': '20px', 'font-size': '56px'}
                ),

                html.Div(
                    className='form-container',
                    style={
                        'background-color': 'floralwhite',
                        'padding': '20px',
                        'border-radius': '5px',
                        'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
                        'margin-bottom': '20px'
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Label('House Number (1-1000): '),
                                dcc.Input(id='input-house', type='number', min=1, max=1000, placeholder='Enter house number'),
                            ]
                        ),
                        html.Br(),
                        html.Div(
    children=[
        html.Label('Appliance Name: '),
        dcc.Dropdown(
            id='input-appliance',
            options=[
                {'label': 'Air Conditioner', 'value': 'Air_Conditioner [kW]'},
                {'label': 'Fans', 'value': 'Fans [kW]'},
                {'label': 'Tubelight', 'value': 'Tubelight [kW]'},
                {'label': 'Television', 'value': 'Television [kW]'},
                {'label': 'Washing Machine', 'value': 'Washing_Machine [kW]'},
                {'label': 'Water Geyser', 'value': 'Water_Geyser [kW]'},
                {'label': 'Mixer Grinder', 'value': 'Mixer_Grinder [kW]'},
                {'label': 'Wet Grinder', 'value': 'Wet_Grinder [kW]'},
                {'label': 'Refrigerator', 'value': 'Refrigerator [kW]'},
                {'label': 'Dishwasher', 'value': 'Dishwasher [kW]'},
                {'label': 'Microwave Oven', 'value': 'Microwave_Oven [kW]'},
            ],
            placeholder='Select appliance'
        ),
    ]
),
                        html.Br(),
                        html.Div(
                            className='query-container',
                            style={
                                'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center',
                                'margin-bottom': '10px'
                            },
                            children=[
                                html.Label('Query: '),
                                dcc.Input(id='input-query', type='text', placeholder='Enter your query', className='query-input'),
                                html.Button('Submit Query', id='button-query', n_clicks=0, style={'background-color': '#007BFF', 'color': 'white'})
                            ]
                        ),
                        html.Div(
                            className='buttons-container',
                            style={'text-align': 'center', 'margin-top': '20px'},
                            children=[
                                html.Button('Total Power Consumption', id='button-consumption', n_clicks=0, style={'background-color': '#28A745', 'color': 'white'}),
                                html.Button('Total Cost', id='button-cost', n_clicks=0, style={'background-color': '#DC3545', 'color': 'white'}),
                                html.Button('Power Consumption Percentage', id='button-percentage', n_clicks=0, style={'background-color': '#FFC107', 'color': 'white'}),
                                html.Button('Peak Hours', id='button-peak-hours', n_clicks=0, style={'background-color': '#17A2B8', 'color': 'white'}),
                                html.Button('Off-Peak Hours', id='button-off-peak-hours', n_clicks=0, style={'background-color': '#6610F2', 'color': 'white'}),
                                html.Button('Power Saving Tips', id='button-saving-tips', n_clicks=0, style={'background-color': '#6C757D', 'color': 'white'}),
                            ]
                        )
                    ]
                ),

                html.Div(id='output-prediction', className='output-container'),
            ]
        )
    ]
)


@app.callback(
    Output('output-prediction', 'children'),
    [Input('button-consumption', 'n_clicks'),
     Input('button-cost', 'n_clicks'),
     Input('button-percentage', 'n_clicks'),
     Input('button-peak-hours', 'n_clicks'),
     Input('button-off-peak-hours', 'n_clicks'),
     Input('button-saving-tips', 'n_clicks'),
     Input('button-query', 'n_clicks')],
    [State('input-house', 'value'),
     State('input-appliance', 'value'),
     State('input-query', 'value')]
)
def update_output(btn_consumption, btn_cost, btn_percentage, btn_peak_hours, btn_off_peak_hours, btn_saving_tips, btn_query, house, appliance, query):
    if house is None or house not in range(1, 1001):
        return "Please enter a valid house number between 1 and 1000."
    
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'button-query' and query:
        query_lower = query.lower()
        if "total power consumption" in query_lower:
            return f"House {house}: Total predicted power consumption for the next 60 days is {total_predicted_consumption_house_kwh[house]:.2f} kWh."
        elif "total cost" in query_lower:
            cost, explanation = calculate_bill(total_predicted_consumption_house_kwh[house])
            return f"House {house}: Total predicted cost for the next 60 days is ₹{cost:.2f}. {explanation}"
        elif "power consumption percentage" in query_lower:
            if appliance is None:
                return "Please enter an appliance name."
            if appliance not in predictions[house]:
                return f"Appliance '{appliance}' not found in House {house}."
            appliance_predicted_kwh = predictions[house][appliance].sum()
            total_house_kwh = total_predicted_consumption_house_kwh[house]
            if total_house_kwh > 0:
                percentage = (appliance_predicted_kwh / total_house_kwh) * 100
                return f"House {house}: {appliance} is predicted to consume {appliance_predicted_kwh:.2f} kWh, which is {percentage:.2f}% of the total consumption for the next 60 days."
            return f"House {house}: Total power consumption for this house is zero. Cannot calculate percentage."
        elif "peak hours" in query_lower:
            if appliance is None:
                return "Please enter an appliance name."
            if appliance not in peak_hours_house.get(house, {}):
                return f"Appliance '{appliance}' not found in House {house}."
            return f"House {house}: Peak hours for {appliance} are {peak_hours_house[house][appliance]}."
        elif "off peak hours" in query_lower:
            if appliance is None:
                return "Please enter an appliance name."
            if appliance not in off_peak_hours_house.get(house, {}):
                return f"Appliance '{appliance}' not found in House {house}."
            return f"House {house}: Off-peak hours for {appliance} are {off_peak_hours_house[house][appliance]}."
        elif "power saving tips" in query_lower:
            if appliance is None:
                return "Please enter an appliance name."
            if appliance not in peak_hours_house.get(house, {}) or appliance not in off_peak_hours_house.get(house, {}):
                return f"Appliance '{appliance}' not found in House {house}."
            peak_hours = peak_hours_house[house][appliance]
            off_peak_hours = off_peak_hours_house[house][appliance]
            return f"House {house}: To save power for {appliance}, use it less during peak hours {peak_hours} and more during off-peak hours {off_peak_hours}."
        else:
            return "Please enter a valid query."

    if button_id == 'button-consumption':
        return f"House {house}: Total predicted power consumption for the next 60 days is {total_predicted_consumption_house_kwh[house]:.2f} kWh."
    
    elif button_id == 'button-cost':
        cost, explanation = calculate_bill(total_predicted_consumption_house_kwh[house])
        return f"House {house}: Total predicted cost for the next 60 days is ₹{cost:.2f}. {explanation}"

    elif button_id == 'button-percentage':
        if appliance is None:
            return "Please enter an appliance name."
        if appliance not in predictions[house]:
            return f"Appliance '{appliance}' not found in House {house}."
        appliance_predicted_kwh = predictions[house][appliance].sum()
        total_house_kwh = total_predicted_consumption_house_kwh[house]
        if total_house_kwh > 0:
            percentage = (appliance_predicted_kwh / total_house_kwh) * 100
            return f"House {house}: {appliance} is predicted to consume {appliance_predicted_kwh:.2f} kWh, which is {percentage:.2f}% of the total consumption for the next 60 days."
        return f"House {house}: Total power consumption for this house is zero. Cannot calculate percentage."
    

        
    elif button_id == 'button-peak-hours':
        if appliance is None:
            return "Please enter an appliance name."
        if appliance not in peak_hours_house.get(house, {}):
            return f"Appliance '{appliance}' not found in House {house}."
        return f"House {house}: Peak hours for {appliance} are {peak_hours_house[house][appliance]}."

    elif button_id == 'button-off-peak-hours':
        if appliance is None:
            return "Please enter an appliance name."
        if appliance not in off_peak_hours_house.get(house, {}):
            return f"Appliance '{appliance}' not found in House {house}."
        return f"House {house}: Off-peak hours for {appliance} are {off_peak_hours_house[house][appliance]}."

    
    elif button_id == 'button-saving-tips':
        if appliance is None:
            return "Please enter an appliance name."
        if appliance not in peak_hours_house.get(house, {}) or appliance not in off_peak_hours_house.get(house, {}):
            return f"Appliance '{appliance}' not found in House {house}."
        peak_hours = peak_hours_house[house][appliance]
        off_peak_hours = off_peak_hours_house[house][appliance]
        return f"House {house}: To save power for {appliance}, use it less during peak hours {peak_hours} and more during off-peak hours {off_peak_hours}."
    
    return "Please enter a valid query."


if __name__ == '__main__':
    app.run_server(debug=True,port=8051) 