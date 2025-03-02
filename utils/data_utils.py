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
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))

        # Add derived features
        data['Load_Fuel_Efficiency'] = data['Load Factor (%)'] * data['Fuel Efficiency (ASK)']
        data['Profit per ASK'] = data['Revenue per ASK'] - data['Cost per ASK']
        data['Ancillary_Revenue_Ratio'] = data['Ancillary Revenue (USD)'] / data['Revenue (USD)']
        data['Real Revenue'] = data['Revenue (USD)'] - data['Operating Cost (USD)']
        data['Load Utilization'] = data['Load Factor (%)'] * data['Aircraft Utilization (Hours/Day)']

        # Process time-related data if available
        try:
            data['Scheduled Departure Time'] = pd.to_datetime(data['Scheduled Departure Time'])
            data['Actual Departure Time'] = pd.to_datetime(data['Actual Departure Time'])
            data['Delay (Minutes)'] = (data['Actual Departure Time'] - data['Scheduled Departure Time']).dt.total_seconds() / 60
            data['Departure Hour'] = data['Scheduled Departure Time'].dt.hour
            data['Day of Week'] = data['Scheduled Departure Time'].dt.dayofweek
        except:
            print("Time-related columns not available or could not be processed")

        # Fill missing values
        data = data.fillna(data.median())

        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_feature_groups():
    """
    Return grouped features for analysis
    """
    return {
        'Revenue Metrics': [
            'Revenue (USD)',
            'Revenue per ASK',
            'Ancillary Revenue (USD)',
            'Ancillary_Revenue_Ratio',
            'Real Revenue'
        ],
        'Operational Metrics': [
            'Load Factor (%)',
            'Aircraft Utilization (Hours/Day)',
            'Fleet Availability (%)',
            'Turnaround Time (Minutes)',
            'Load Utilization'
        ],
        'Financial Metrics': [
            'Operating Cost (USD)',
            'Cost per ASK',
            'Profit (USD)',
            'Profit per ASK',
            'Net Profit Margin (%)',
            'Debt-to-Equity Ratio'
        ],
        'Efficiency Metrics': [
            'Fuel Efficiency (ASK)',
            'Load_Fuel_Efficiency',
            'Delay (Minutes)'
        ]
    }

def preprocess_data(data):
    """
    Preprocess the data for model training.
    """
    if data is None:
        raise ValueError("No data available for preprocessing")

    feature_groups = get_feature_groups()
    all_features = []
    for group in feature_groups.values():
        all_features.extend(group)

    # Remove duplicates while preserving order
    feature_cols = list(dict.fromkeys(all_features))

    # Remove target variable from features if present
    target_col = 'Profit (USD)'
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # Keep only available columns
    feature_cols = [col for col in feature_cols if col in data.columns]

    X = data[feature_cols]
    y = data[target_col]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y