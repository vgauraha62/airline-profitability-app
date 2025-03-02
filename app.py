import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import initialize_models, train_model, predict
from utils.plot_utils import create_plot
from utils.data_utils import load_data, preprocess_data, get_feature_groups
import plotly.graph_objects as go

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
                model_metrics.append({
                    'Model': selected_model,
                    'R² Score': metrics['r2'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse']
                })

                # Generate predictions for later export
                predictions = model.predict(X[selected_features])
                st.session_state.current_predictions = predictions

            except Exception as model_error:
                st.error(f"Error in {selected_model} training: {str(model_error)}")
                continue

        # Display model comparison
        if model_metrics:
            metrics_df = pd.DataFrame(model_metrics)
            fig = go.Figure(data=[
                go.Bar(name=metric, x=metrics_df['Model'], y=metrics_df[metric])
                for metric in ['R² Score', 'MAE', 'RMSE']
            ])
            fig.update_layout(
                title='Model Performance Comparison',
                barmode='group',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Aviation insights based on model performance
            best_model = metrics_df.loc[metrics_df['R² Score'].idxmax(), 'Model']
            st.info(f"""
            💡 **Model Performance Insights:**
            - {best_model} shows the best performance with an R² score of {metrics_df['R² Score'].max():.3f}
            - This indicates that {metrics_df['R² Score'].max()*100:.1f}% of the variance in profit can be explained by the selected features
            - The model's predictions have an average error of ${metrics_df['MAE'].min():,.2f}
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

if __name__ == "__main__":
    main()