import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import initialize_models, train_model, predict
from utils.plot_utils import create_plot
from utils.data_utils import load_data, preprocess_data

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

def main():
    st.title("Aviation Revenue ML Model Visualization")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load and preprocess data
    try:
        data = load_data()
        X, y = preprocess_data(data)
        
        # Feature selection
        available_features = X.columns.tolist()
        selected_features = st.sidebar.multiselect(
            "Select Features",
            available_features,
            default=available_features[:3]
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
            "Learning Curve"
        ]
        selected_plot = st.sidebar.selectbox(
            "Select Plot Type",
            plot_options
        )
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Visualization")
            if selected_features and selected_model:
                # Train model if needed
                if model_options[selected_model] not in st.session_state.trained_models:
                    with st.spinner("Training model..."):
                        model = train_model(
                            X[selected_features],
                            y,
                            model_type=model_options[selected_model]
                        )
                        st.session_state.trained_models[model_options[selected_model]] = model
                
                # Create and display plot
                fig = create_plot(
                    st.session_state.trained_models[model_options[selected_model]],
                    X[selected_features],
                    y,
                    plot_type=selected_plot
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Model Performance")
            if selected_features and selected_model:
                model = st.session_state.trained_models[model_options[selected_model]]
                
                # Display model metrics
                metrics = model.get_metrics()
                st.metric("R² Score", f"{metrics['r2']:.3f}")
                st.metric("MAE", f"{metrics['mae']:.3f}")
                st.metric("RMSE", f"{metrics['rmse']:.3f}")
                
                # Prediction interface
                st.subheader("Make Predictions")
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
                    st.success(f"Predicted Revenue: ${prediction[0]:,.2f}")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data and try again.")

if __name__ == "__main__":
    main()
