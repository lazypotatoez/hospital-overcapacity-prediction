import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

# Load pre-trained model
@st.cache_resource
def load_prediction_model():
    model = load_model('best_deep_model.h5')
    return model

# Prediction function
def predict_capacity(model, input_features):
    input_data = np.array(input_features).reshape(1, -1)  # Reshape to (1, 40)
    prediction = model.predict(input_data)
    return prediction[0][0]  # Return the single predicted value

# Streamlit UI
st.title("Hospital Capacity Prediction & Visualization")

# Year input for prediction
prediction_year = st.number_input("Enter Year to Predict (e.g., 2025)", min_value=2025, step=1)

# Historical data for comparison
# Replace this with your actual data
historical_data = {
    "Year": [2020, 2021, 2022, 2023, 2024],
    "Actual Capacity": [500, 520, 540, 560, 580],  # Example values
}

# Default features (40 features expected by the model)
default_features = [0] * 40  # Replace with actual values used during training

if prediction_year:
    model = load_prediction_model()
    try:
        # Predict capacity for the specified year
        predicted_capacity = predict_capacity(model, default_features)

        # Display prediction results
        st.subheader(f"Predicted Hospital Capacity for {prediction_year}")
        st.write(f"**Predicted Capacity:** {predicted_capacity:.2f}")

        # Combine actual data and prediction into one DataFrame
        years = historical_data["Year"] + [prediction_year]
        actual_capacities = historical_data["Actual Capacity"] + [None]
        predicted_capacities = [None] * len(historical_data["Year"]) + [predicted_capacity]

        # Create a DataFrame for visualization
        data = pd.DataFrame({
            "Year": years,
            "Actual Capacity": actual_capacities,
            "Predicted Capacity": predicted_capacities,
        })

        # Visualization: Actual vs Predicted
        st.subheader("Actual vs Predicted Capacity")
        fig = px.line(
            data_frame=data,
            x="Year",
            y=["Actual Capacity", "Predicted Capacity"],
            labels={"value": "Hospital Capacity", "variable": "Type"},
            title="Actual vs Predicted Hospital Capacity",
            markers=True
        )
        st.plotly_chart(fig)

        # Bar chart for clearer comparison
        st.subheader("Comparison by Year")
        fig_bar = px.bar(
            data_frame=data.melt(id_vars="Year", var_name="Type", value_name="Capacity"),
            x="Year",
            y="Capacity",
            color="Type",
            barmode="group",
            title="Actual vs Predicted Capacity by Year"
        )
        st.plotly_chart(fig_bar)

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
