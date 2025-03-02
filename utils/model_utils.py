import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

class BaseModel:
    def __init__(self):
        self.model = None
        self.metrics = {}
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        self.metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_metrics(self):
        return self.metrics

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

def initialize_models():
    return {
        'lr': LinearRegressionModel(),
        'rf': RandomForestModel(),
        'xgb': XGBoostModel()
    }

def train_model(X, y, model_type):
    models = initialize_models()
    model = models[model_type]
    model.train(X, y)
    return model

def predict(model, X):
    return model.predict(X)
