import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests
import io

def load_data():
    """
    Load the aviation revenue dataset from Google Sheets.
    """
    url = "https://docs.google.com/spreadsheets/d/1eALZhnY5bEJ4uCi9BCjN2fpx8jRIzwWo/export?format=csv&gid=794923645"

    try:
        # Download the CSV data using requests
        response = requests.get(url)
        response.raise_for_status()

        # Read the CSV data into a pandas DataFrame
        data = pd.read_csv(io.StringIO(response.text))

        # Add derived features
        data['Load_Fuel_Efficiency'] = data['Load Factor (%)'] * data['Fuel Efficiency (ASK)']
        data['Profit per ASK'] = data['Revenue per ASK'] - data['Cost per ASK']
        data['Ancillary_Revenue_Ratio'] = data['Ancillary Revenue (USD)'] / data['Revenue (USD)']

        # Drop unnecessary columns
        data = data.drop(columns=['Flight Number', 'Scheduled Departure Time', 'Actual Departure Time'], errors='ignore')

        # Fill missing values
        data = data.fillna(data.median())

        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the data for model training.
    """
    if data is None:
        raise ValueError("No data available for preprocessing")

    # Separate features and target
    target_col = 'Profit (USD)'
    feature_cols = [
        'Load Factor (%)',
        'Revenue per ASK',
        'Cost per ASK',
        'Fuel Efficiency (ASK)',
        'Aircraft Utilization (Hours/Day)',
        'Load_Fuel_Efficiency',
        'Profit per ASK',
        'Ancillary_Revenue_Ratio'
    ]

    X = data[feature_cols]
    y = data[target_col]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y