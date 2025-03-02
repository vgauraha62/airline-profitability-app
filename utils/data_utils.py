import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the aviation revenue dataset.
    Replace this with actual data loading logic.
    """
    # Example data generation for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'passengers': np.random.normal(1000, 200, n_samples),
        'distance': np.random.normal(1500, 300, n_samples),
        'fuel_cost': np.random.normal(500, 100, n_samples),
        'season_factor': np.random.uniform(0.8, 1.2, n_samples),
        'competitor_price': np.random.normal(300, 50, n_samples),
        'revenue': None
    })
    
    # Generate revenue with some realistic relationship
    data['revenue'] = (
        2 * data['passengers'] +
        0.1 * data['distance'] +
        0.5 * data['fuel_cost'] +
        1000 * data['season_factor'] +
        1.5 * data['competitor_price'] +
        np.random.normal(0, 1000, n_samples)
    )
    
    return data

def preprocess_data(data):
    """
    Preprocess the data for model training.
    """
    # Separate features and target
    X = data.drop('revenue', axis=1)
    y = data['revenue']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    return X_scaled, y
