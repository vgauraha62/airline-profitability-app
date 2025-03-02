# Aviation Revenue ML Model Visualization

A Streamlit-based web application for visualizing and analyzing aviation revenue predictions using machine learning models. This project provides interactive visualizations of aviation metrics, model performance, and revenue predictions.

## Features

- **Multiple ML Models**: Support for Linear Regression, Random Forest, and XGBoost models
- **Feature Group Analysis**: Organized feature selection by categories:
  - Revenue Metrics
  - Operational Metrics
  - Financial Metrics
  - Efficiency Metrics
- **Interactive Visualizations**:
  - Feature Importance
  - Prediction vs Actual Values
  - Residual Analysis
  - Learning Curves
  - Correlation Matrices
  - Load Factor Analysis
  - Delay Analysis
- **Real-time Predictions**: Input custom values to get instant profit predictions
- **Performance Metrics**: Detailed model performance visualization including R², MAE, and RMSE

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aviation-revenue-ml.git
cd aviation-revenue-ml
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── utils/
│   ├── data_utils.py     # Data loading and preprocessing
│   ├── model_utils.py    # ML model implementations
│   └── plot_utils.py     # Visualization functions
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Usage

1. Select a feature group from the sidebar
2. Choose specific features for analysis
3. Select a machine learning model
4. Choose visualization type
5. Input custom values for predictions

## Data Description

The application uses aviation data with the following key metrics:
- Load Factor
- Revenue per ASK (Available Seat Kilometer)
- Operating Costs
- Aircraft Utilization
- Fleet Availability
- Fuel Efficiency
- And more...

## Model Performance

The application includes three machine learning models:
1. **Linear Regression**: Basic linear model for revenue prediction
2. **Random Forest**: Ensemble learning model for complex patterns
3. **XGBoost**: Gradient boosting model for high accuracy

## Docker Considerations

While Docker can be used for this project, it's not strictly necessary for the following reasons:

1. **Simple Dependencies**: The project uses standard Python packages that are easy to install
2. **Streamlit Deployment**: Streamlit provides simple deployment options without containerization
3. **Resource Efficiency**: Running without Docker reduces overhead for local development

However, if you want to include Docker for consistency or deployment purposes, you can use this sample Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
