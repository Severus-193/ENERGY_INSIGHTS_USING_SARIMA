from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import re

app = Flask(__name__)

# Directory where CSV files are stored
data_directory = './data'

# Ensure the data directory exists
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Define the date ranges
start_date = '2023-10-16'
end_date = '2024-02-15'
new_daily_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
new_hourly_date_range = pd.date_range(start=start_date, end=end_date, freq='H')

# Define the list of appliances
appliances = [
    "Air_Conditioner [kW]", "Fans [kW]", "Tubelight [kW]", "Television [kW]",
    "Washing_Machine [kW]", "Water_Geyser [kW]", "Mixer_Grinder [kW]",
    "Wet_Grinder [kW]", "Refrigerator [kW]", "Microwave_Oven [kW]", "Dishwasher [kW]"
]

# Function to generate daily data
def generate_daily_data(seed, appliance_info):
    np.random.seed(seed)  # Ensure reproducibility
    daily_data = pd.DataFrame(index=new_daily_date_range)

    for appliance, count in appliance_info.items():
        daily_data[appliance] = (np.random.uniform(0.02, 2.0, len(daily_data)) + np.random.uniform(-0.5, 0.5)) * count

    daily_data['use [kW]'] = daily_data.sum(axis=1)
    daily_data['gen [kW]'] = np.random.uniform(0.1, 1.0, len(daily_data))
    daily_data['House overall [kW]'] = daily_data['use [kW]'] - daily_data['gen [kW]']
    daily_data.loc[daily_data.index.dayofweek == 6, appliance_info.keys()] = 0  # Sundays
    daily_data.loc[daily_data.index.dayofweek == 5, appliance_info.keys()] *= 2  # Saturdays
    daily_data.loc[daily_data.index.dayofweek == 2, appliance_info.keys()] /= 2  # Wednesdays
    daily_data['use [kW]'] = daily_data[appliance_info.keys()].sum(axis=1)
    daily_data['House overall [kW]'] = daily_data['use [kW]'] - daily_data['gen [kW]']
    daily_data['House overall [kW]'] = daily_data['House overall [kW]'].clip(lower=0)
    return daily_data

# Function to generate hourly data
def generate_hourly_data(seed, appliance_info):
    np.random.seed(seed)  # Ensure reproducibility
    hourly_data = pd.DataFrame(index=new_hourly_date_range)

    for appliance, count in appliance_info.items():
        hourly_data[appliance] = (np.random.uniform(0.05, 0.3, len(hourly_data)) + np.random.uniform(-0.05, 0.05)) * count

    hourly_data['use [kW]'] = hourly_data.sum(axis=1)
    hourly_data['gen [kW]'] = np.random.uniform(0.01, 0.1, len(hourly_data))
    hourly_data['House overall [kW]'] = hourly_data['use [kW]'] - hourly_data['gen [kW]']
    hourly_data['House overall [kW]'] = hourly_data['House overall [kW]'].clip(lower=0)
    return hourly_data

# Function to determine the next house number
def get_next_house_number():
    files = os.listdir(data_directory)
    house_numbers = [int(re.search(r'\d+', f).group()) for f in files if re.search(r'\d+', f)]
    return max(house_numbers, default=8) + 1

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Generate a new house number sequentially
        house_number = get_next_house_number()

        # Get appliance counts from the form
        appliance_info = {appliance: int(request.form[appliance]) for appliance in appliances}

        # Generate datasets for the given house number
        daily_data = generate_daily_data(seed=house_number, appliance_info=appliance_info)
        hourly_data = generate_hourly_data(seed=house_number, appliance_info=appliance_info)

        # Save the datasets
        daily_data.to_csv(f'{data_directory}/daily_data_house{house_number}.csv', index_label='time')
        hourly_data.to_csv(f'{data_directory}/hourly_data_house{house_number}.csv', index_label='time')

        return render_template('index.html', appliances=appliances, house_number=house_number,
                               daily_data_sample=daily_data.head().to_html(classes='data'),
                               hourly_data_sample=hourly_data.head().to_html(classes='data'))

    return render_template('index.html', appliances=appliances)

if __name__ == '__main__':
    app.run(debug=True)
