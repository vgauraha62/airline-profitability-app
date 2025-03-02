# Analitica Global - Aviation Dataset Insights

A comprehensive machine learning application for aviation revenue prediction and model visualization, designed to provide actionable insights into aviation metrics and profitability analysis.

## 🚀 Features

- **Advanced Model Comparison**: Compare multiple ML models (Linear Regression, Random Forest, XGBoost)
  - Cross-validation scores and model stability analysis
  - Feature importance comparison across models
  - Comprehensive performance metrics visualization
  - Error distribution and prediction accuracy analysis
  - Model-specific strengths and weaknesses

- **Interactive Visualizations**:
  - Revenue correlation analysis
  - Load factor impact assessment
  - Feature importance comparison
  - Performance metrics dashboards
  - Prediction error distribution

- **Comprehensive Analytics**:
  - Revenue Metrics (Revenue per ASK, Ancillary Revenue)
  - Operational Metrics (Load Factor, Fleet Availability)
  - Financial Metrics (Operating Costs, Profit Margins)
  - Efficiency Metrics (Fuel Efficiency, Asset Utilization)

- **Business Intelligence**:
  - Automated insights generation
  - Aviation-specific recommendations
  - Profitability analysis
  - Performance optimization suggestions

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aviation-analytics.git
cd aviation-analytics
```

2. Create and activate a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install streamlit==1.42.2
pip install pandas==2.2.3
pip install numpy==2.2.3
pip install scikit-learn==1.6.1
pip install xgboost==2.1.4
pip install plotly==6.0.0
```

4. Run the application:
```bash
streamlit run app.py
```

## 📊 Model Comparison Features

### Performance Metrics
- R² Score (Model Accuracy)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Cross-validation Scores
- Feature Importance Rankings
- Prediction Error Distribution

### Available Models
- Linear Regression (Baseline)
- Random Forest (Ensemble Learning)
- XGBoost (Gradient Boosting)

### Aviation Metrics Analyzed
- Load Factor (%)
- Revenue per ASK
- Operating Costs
- Aircraft Utilization
- Fleet Availability
- Fuel Efficiency
- Ancillary Revenue Ratio


## 📈 Usage Guide

1. **Data Selection**:
   - Choose feature groups from the sidebar
   - Select specific features for analysis
   - Multiple feature selection supported

2. **Model Analysis**:
   - Select multiple models for comparison
   - View comprehensive performance metrics
   - Analyze feature importance across models

3. **Visualization Options**:
   - Performance comparison charts
   - Feature correlation matrices
   - Error distribution analysis
   - Load factor impact studies

4. **Export and Documentation**:
   - Download analysis results
   - Export predictions and metrics
   - Save visualization plots

## 🎯 Project Structure

```
├── app.py                 # Main Streamlit application
├── utils/
│   ├── data_utils.py     # Data loading and preprocessing
│   ├── model_utils.py    # ML model implementations
│   └── plot_utils.py     # Visualization functions
├── .streamlit/
│   └── config.toml       # Streamlit configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🛫 Production Deployment

This application can be deployed using:
1. **Streamlit Cloud** (Recommended):
   - Easy deployment with direct GitHub integration
   - Automatic updates with repository changes
   - Free tier available for personal projects

2. **Local Server**:
   - Run using Streamlit's built-in server
   - Accessible via localhost:5000
   - Suitable for development and testing

Docker containerization is optional but can be useful for consistent deployment environments.

## ✨ Future Enhancements

- Additional machine learning models
- Real-time data integration
- Advanced feature engineering
- Custom model parameter tuning
- Automated report generation
- Time series analysis capabilities

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)