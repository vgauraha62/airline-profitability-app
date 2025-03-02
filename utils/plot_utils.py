import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.model_selection import learning_curve
import pandas as pd

def create_plot(model, X, y, plot_type, raw_data=None):
    if plot_type == "Feature Importance":
        return plot_feature_importance(model, X)
    elif plot_type == "Prediction vs Actual":
        return plot_prediction_vs_actual(model, X, y)
    elif plot_type == "Residual Plot":
        return plot_residuals(model, X, y)
    elif plot_type == "Learning Curve":
        return plot_learning_curve(model, X, y)
    elif plot_type == "Correlation Matrix":
        return plot_correlation_matrix(raw_data)
    elif plot_type == "Load Factor vs Profit":
        return plot_load_factor_profit(raw_data)
    elif plot_type == "Revenue Metrics":
        return plot_revenue_metrics(raw_data)

def plot_feature_importance(model, X):
    if hasattr(model.model, 'feature_importances_'):
        importance = model.model.feature_importances_
    else:
        importance = np.abs(model.model.coef_)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=X.columns,
        y=importance,
        name='Feature Importance'
    ))
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        template='plotly_white'
    )
    return fig

def plot_prediction_vs_actual(model, X, y):
    predictions = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y,
        y=predictions,
        mode='markers',
        name='Predictions'
    ))

    # Add diagonal line
    min_val = min(min(y), min(predictions))
    max_val = max(max(y), max(predictions))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='Predicted vs Actual Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        template='plotly_white'
    )
    return fig

def plot_correlation_matrix(data):
    corr_matrix = data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title='Correlation Matrix',
        width=800,
        height=800,
        template='plotly_white'
    )
    return fig

def plot_load_factor_profit(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Load Factor (%)'],
        y=data['Profit (USD)'],
        mode='markers',
        name='Data Points'
    ))

    fig.update_layout(
        title='Load Factor vs Profit',
        xaxis_title='Load Factor (%)',
        yaxis_title='Profit (USD)',
        template='plotly_white'
    )
    return fig

def plot_revenue_metrics(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Revenue per ASK'],
        y=data['Profit (USD)'],
        mode='markers',
        name='Revenue per ASK vs Profit'
    ))

    fig.update_layout(
        title='Revenue Metrics Analysis',
        xaxis_title='Revenue per ASK',
        yaxis_title='Profit (USD)',
        template='plotly_white'
    )
    return fig

def plot_residuals(model, X, y):
    predictions = model.predict(X)
    residuals = y - predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions,
        y=residuals,
        mode='markers',
        name='Residuals'
    ))

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        template='plotly_white'
    )
    return fig

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model.model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()

    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue')
    ))

    # Test scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_mean,
        mode='lines+markers',
        name='Cross-validation Score',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training Examples',
        yaxis_title='Score',
        template='plotly_white'
    )
    return fig