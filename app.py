import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import initialize_models, train_model, predict
from utils.plot_utils import create_plot
from utils.data_utils import load_data, preprocess_data, get_feature_groups

# Page configuration
st.set_page_config(
    page_title="Aviation Revenue ML Model Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

def main():
    st.title("Aviation Revenue ML Model Visualization")

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
        selected_model = st.sidebar.selectbox(
            "Select Model",
            list(model_options.keys())
        )

        # Plot selection
        plot_options = [
            "Feature Importance",
            "Prediction vs Actual",
            "Residual Plot",
            "Learning Curve",
            "Performance Metrics",
            "Feature Correlation Matrix",
            "Revenue Correlation",
            "Load Factor Analysis",
            "Delay Analysis"
        ]
        selected_plot = st.sidebar.selectbox(
            "Select Plot Type",
            plot_options
        )

        # Main content
        if not selected_features:
            st.warning("Please select at least one feature to begin analysis.")
            return

        # Performance Metrics Section at the top
        st.header("Model Performance Metrics")
        metrics_cols = st.columns(3)

        # Initialize metrics as None
        metrics = None

        if selected_features and selected_model:
            # Train model if needed
            model_key = f"{model_options[selected_model]}_{','.join(selected_features)}"

            try:
                if model_key not in st.session_state.trained_models:
                    with st.spinner("Training model..."):
                        model = train_model(
                            X[selected_features],
                            y,
                            model_type=model_options[selected_model]
                        )
                        st.session_state.trained_models[model_key] = model

                model = st.session_state.trained_models[model_key]
                metrics = model.get_metrics()

                # Display metrics in large format
                with metrics_cols[0]:
                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                    st.markdown(f"""
                        <div style='text-align: center; font-size: 0.8em;'>
                            Explains {metrics['r2']*100:.1f}% of variance
                        </div>
                        """, unsafe_allow_html=True)

                with metrics_cols[1]:
                    st.metric("Mean Absolute Error", f"{metrics['mae']:.3f}")
                    st.markdown(f"""
                        <div style='text-align: center; font-size: 0.8em;'>
                            Average prediction error
                        </div>
                        """, unsafe_allow_html=True)

                with metrics_cols[2]:
                    st.metric("Root Mean Squared Error", f"{metrics['rmse']:.3f}")
                    st.markdown(f"""
                        <div style='text-align: center; font-size: 0.8em;'>
                            Standard deviation of prediction errors
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as model_error:
                st.error(f"Error in model training: {str(model_error)}")
                return

        # Visualization and Prediction sections
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Model Visualization")
            if metrics is not None:  # Only show plot if model is trained
                try:
                    fig = create_plot(
                        model,
                        X[selected_features],
                        y,
                        plot_type=selected_plot,
                        raw_data=data
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as plot_error:
                    st.error(f"Error creating plot: {str(plot_error)}")

        with col2:
            st.subheader("Make Predictions")
            if metrics is not None:  # Only show prediction interface if model is trained
                try:
                    input_data = {}
                    for feature in selected_features:
                        input_data[feature] = st.number_input(
                            f"Enter {feature}",
                            value=float(X[feature].mean()),
                            format="%.2f"
                        )

                    if st.button("Predict"):
                        prediction = predict(
                            model,
                            pd.DataFrame([input_data])
                        )
                        st.success(f"Predicted Profit: ${prediction[0]:,.2f}")
                except Exception as pred_error:
                    st.error(f"Error making prediction: {str(pred_error)}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()