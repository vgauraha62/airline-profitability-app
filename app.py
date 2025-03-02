import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import initialize_models, train_model, predict
from utils.plot_utils import create_plot
from utils.data_utils import load_data, preprocess_data, get_feature_groups
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Analitica Global - Aviation Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1em;
    }
    .sub-header {
        font-size: 1.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2em;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

def export_data(data, predictions, features):
    """Export data and predictions to CSV"""
    export_df = pd.DataFrame({
        'Actual': data['Profit (USD)'],
        'Predicted': predictions
    })
    for feature in features:
        export_df[feature] = data[feature]
    return export_df

def main():
    # Headers
    st.markdown('<h1 class="main-header">Analitica Global</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">HACKATHON - AVIATION DATASET INSIGHTS</h2>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Configuration")

    try:
        # Load data if not already loaded
        if st.session_state.raw_data is None:
            with st.spinner("Loading data..."):
                data = load_data()
                if data is not None:
                    st.session_state.raw_data = data
                else:
                    st.error("Failed to load data. Please check the data source.")
                    return

        data = st.session_state.raw_data
        X, y = preprocess_data(data)

        # Feature group selection
        feature_groups = get_feature_groups()
        selected_group = st.sidebar.selectbox(
            "Select Feature Group",
            ["All Features"] + list(feature_groups.keys())
        )

        if selected_group == "All Features":
            available_features = X.columns.tolist()
        else:
            available_features = [f for f in feature_groups[selected_group] if f in X.columns]

        selected_features = st.sidebar.multiselect(
            "Select Features",
            available_features,
            default=available_features[:3] if available_features else []
        )

        # Model selection
        model_options = {
            "Linear Regression": "lr",
            "Random Forest": "rf",
            "XGBoost": "xgb"
        }

        # Allow multiple model selection for comparison
        selected_models = st.sidebar.multiselect(
            "Select Models to Compare",
            list(model_options.keys()),
            default=[list(model_options.keys())[0]]
        )

        if not selected_features:
            st.warning("Please select at least one feature to begin analysis.")
            return

        # Model Comparison Section
        st.header("Model Comparison")
        model_metrics = []
        feature_importance_data = {}
        cv_scores = {}

        for selected_model in selected_models:
            model_key = f"{model_options[selected_model]}_{','.join(selected_features)}"

            try:
                if model_key not in st.session_state.trained_models:
                    with st.spinner(f"Training {selected_model}..."):
                        model = train_model(
                            X[selected_features],
                            y,
                            model_type=model_options[selected_model]
                        )
                        st.session_state.trained_models[model_key] = model

                model = st.session_state.trained_models[model_key]
                metrics = model.get_metrics()

                # Cross-validation scores
                cv_scores[selected_model] = cross_val_score(
                    model.model, X[selected_features], y, cv=5, scoring='r2'
                )

                # Feature importance
                if hasattr(model.model, 'feature_importances_'):
                    importance = model.model.feature_importances_
                else:
                    importance = np.abs(model.model.coef_)
                feature_importance_data[selected_model] = dict(zip(selected_features, importance))

                # Expanded metrics
                model_metrics.append({
                    'Model': selected_model,
                    'R² Score': metrics['r2'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'CV R² (mean)': cv_scores[selected_model].mean(),
                    'CV R² (std)': cv_scores[selected_model].std()
                })

                predictions = model.predict(X[selected_features])
                st.session_state.current_predictions = predictions

            except Exception as model_error:
                st.error(f"Error in {selected_model} training: {str(model_error)}")
                continue

        if model_metrics:
            # Create subplots for comprehensive comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Performance Metrics Comparison',
                    'Feature Importance by Model',
                    'Cross-validation Scores Distribution',
                    'Prediction Error Distribution'
                )
            )

            # 1. Performance Metrics Bar Chart
            metrics_df = pd.DataFrame(model_metrics)
            for metric in ['R² Score', 'MAE', 'RMSE']:
                fig.add_trace(
                    go.Bar(name=metric, x=metrics_df['Model'], y=metrics_df[metric]),
                    row=1, col=1
                )

            # 2. Feature Importance Comparison
            for model_name, importances in feature_importance_data.items():
                fig.add_trace(
                    go.Bar(
                        name=model_name,
                        x=list(importances.keys()),
                        y=list(importances.values())
                    ),
                    row=1, col=2
                )

            # 3. Cross-validation Scores
            for model_name, scores in cv_scores.items():
                fig.add_trace(
                    go.Box(name=model_name, y=scores, boxpoints='all'),
                    row=2, col=1
                )

            # 4. Prediction Error Distribution
            for selected_model in selected_models:
                model = st.session_state.trained_models[f"{model_options[selected_model]}_{','.join(selected_features)}"]
                predictions = model.predict(X[selected_features])
                errors = y - predictions
                fig.add_trace(
                    go.Histogram(
                        name=selected_model,
                        x=errors,
                        opacity=0.7,
                        nbinsx=30
                    ),
                    row=2, col=2
                )

            # Update layout with better spacing and readability
            fig.update_layout(
                height=1000,  # Increased height
                title_text="Comprehensive Model Comparison",
                showlegend=True,
                template='plotly_white',
                margin=dict(t=150),  # More space for titles
                grid=dict(rows=2, columns=2, pattern='independent'),
                annotations=[
                    dict(
                        x=ann['x'], y=ann['y'],
                        text=ann['text'],
                        font=dict(size=14),  # Increased font size
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        xanchor='center',
                        yanchor='bottom'
                    ) for ann in fig.layout.annotations
                ]
            )

            # Update subplot spacing
            fig.update_layout(
                height=1000,
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                template='plotly_white'
            )

            # Adjust subplot margins
            fig.update_layout(
                margin=dict(t=150, b=150),
                grid=dict(rows=2, columns=2, pattern='independent'),
                subplot_titles=[
                    '<b>Performance Metrics Comparison</b>',
                    '<b>Feature Importance by Model</b>',
                    '<b>Cross-validation Scores Distribution</b>',
                    '<b>Prediction Error Distribution</b>'
                ]
            )


            st.plotly_chart(fig, use_container_width=True)

            # Detailed insights
            st.subheader("Model Performance Analysis")

            # Best model identification
            best_r2_model = metrics_df.loc[metrics_df['R² Score'].idxmax()]
            best_mae_model = metrics_df.loc[metrics_df['MAE'].idxmin()]

            st.info(f"""
            💡 **Key Model Insights:**

            Best Overall Performance:
            - {best_r2_model['Model']} achieves the highest R² score of {best_r2_model['R² Score']:.3f}
            - This means it explains {best_r2_model['R² Score']*100:.1f}% of the variance in profit predictions

            Prediction Accuracy:
            - {best_mae_model['Model']} has the lowest Mean Absolute Error of ${best_mae_model['MAE']:,.2f}
            - Cross-validation shows model stability with R² variation of ±{best_r2_model['CV R² (std)']:.3f}

            Feature Impact Analysis:
            {get_feature_impact_insights(feature_importance_data)}
            """)

            # Aviation-specific insights
            st.info(f"""
            💡 **Aviation Business Insights:**

            Revenue Optimization:
            - The model suggests focusing on {get_top_features(feature_importance_data)} for maximum profit impact
            - Consider these factors when planning route optimization and pricing strategies

            Operational Recommendations:
            - Based on feature importance, prioritize {get_operational_recommendations(feature_importance_data)}
            - Monitor these metrics closely for better profit prediction and optimization
            """)

        # Feature Insights
        st.header("Feature Analysis")
        if selected_features:
            correlations = data[selected_features + ['Profit (USD)']].corr()['Profit (USD)'].sort_values(ascending=False)

            st.write("**Key Feature Insights:**")
            for feature, corr in correlations.items():
                if feature != 'Profit (USD)':
                    impact = "strong positive" if corr > 0.7 else "moderate positive" if corr > 0.3 else "weak positive" if corr > 0 else "strong negative" if corr < -0.7 else "moderate negative" if corr < -0.3 else "weak negative"
                    st.write(f"- {feature} has a {impact} correlation ({corr:.2f}) with profitability")

        # Export Options
        st.header("Export Options")
        if st.session_state.current_predictions is not None:
            export_data_df = export_data(data, st.session_state.current_predictions, selected_features)
            csv = export_data_df.to_csv(index=False)
            st.download_button(
                label="Download Analysis Results",
                data=csv,
                file_name="aviation_analysis_results.csv",
                mime="text/csv"
            )

        # Visualization Section
        st.header("Detailed Visualizations")
        plot_options = [
            "Feature Importance",
            "Prediction vs Actual",
            "Residual Plot",
            "Learning Curve",
            "Performance Metrics",
            "Feature Correlation Matrix",
            "Revenue Correlation",
            "Load Factor Analysis"
        ]
        selected_plot = st.selectbox("Select Visualization", plot_options)

        if len(selected_models) > 0:
            # Use the first selected model for visualization
            model_key = f"{model_options[selected_models[0]]}_{','.join(selected_features)}"
            model = st.session_state.trained_models[model_key]

            try:
                fig = create_plot(
                    model,
                    X[selected_features],
                    y,
                    plot_type=selected_plot,
                    raw_data=data
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add relevant aviation insights based on plot type
                if selected_plot == "Load Factor Analysis":
                    st.info("""
                    💡 **Load Factor Insights:**
                    - Higher load factors generally correlate with improved profitability
                    - Optimal load factor targeting can significantly impact revenue
                    - Consider seasonal variations in load factor planning
                    """)
                elif selected_plot == "Revenue Correlation":
                    st.info("""
                    💡 **Revenue Insights:**
                    - Strong correlation between ancillary revenue and overall profitability
                    - Operating costs show significant impact on net margins
                    - Focus on revenue per ASK optimization for better performance
                    """)

            except Exception as plot_error:
                st.error(f"Error creating plot: {str(plot_error)}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

def get_feature_impact_insights(importance_data):
    # Get the average importance across all models for each feature
    avg_importance = {}
    for model_imps in importance_data.values():
        for feature, imp in model_imps.items():
            avg_importance[feature] = avg_importance.get(feature, 0) + imp

    avg_importance = {k: v/len(importance_data) for k, v in avg_importance.items()}
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    insights = []
    for feature, importance in sorted_features[:3]:
        insights.append(f"- {feature} shows consistent importance across models (avg. impact: {importance:.3f})")

    return "\n".join(insights)

def get_top_features(importance_data):
    avg_importance = {}
    for model_imps in importance_data.values():
        for feature, imp in model_imps.items():
            avg_importance[feature] = avg_importance.get(feature, 0) + imp

    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    return ", ".join(f"{feature}" for feature, _ in sorted_features[:2])

def get_operational_recommendations(importance_data):
    operational_features = [
        'Load Factor (%)', 'Aircraft Utilization (Hours/Day)', 
        'Fleet Availability (%)', 'Fuel Efficiency (ASK)'
    ]

    relevant_features = []
    for feature in operational_features:
        if any(feature in model_imps for model_imps in importance_data.values()):
            relevant_features.append(feature)

    return ", ".join(relevant_features[:2])


if __name__ == "__main__":
    main()