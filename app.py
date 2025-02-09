import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import plotly.express as px

# Load pre-trained model (replace 'model.h5' with your model file)
@st.cache_resource
def load_prediction_model():
    model = load_model('best_deep_model.h5')
    return model

# Prediction function
def predict_capacity(model, input_features):
    input_data = np.array(input_features).reshape(1, -1)  # Adjust input shape based on the model
    prediction = model.predict(input_data)
    return prediction[0][0]  # Return the single predicted value

# Streamlit UI
st.title("Hospital Capacity Prediction & Visualization")

# Sidebar for user inputs
st.sidebar.header("User Input Features")
monthly_admissions = st.sidebar.slider("Monthly Hospital Admissions", 0, 1000, 500)
bed_availability = st.sidebar.slider("Bed Availability (%)", 0, 100, 50)
outpatient_visits = st.sidebar.slider("Monthly Outpatient Visits", 0, 5000, 2500)
gov_health_expenditure = st.sidebar.slider("Government Health Expenditure ($M)", 0, 500, 200)

# Year input for prediction
st.sidebar.header("Year for Prediction")
prediction_year = st.sidebar.number_input("Enter Year to Predict (e.g., 2025)", min_value=2025, step=1)

# Combine inputs for prediction
input_features = [monthly_admissions, bed_availability, outpatient_visits, gov_health_expenditure]

# Predict using the model
model = load_prediction_model()
predicted_capacity = predict_capacity(model, input_features)

# Display prediction results
st.subheader(f"Predicted Hospital Capacity for {prediction_year}")
st.write(f"**Predicted Capacity:** {predicted_capacity:.2f}")

# Visualization
st.subheader("Prediction Trend")
# Create a trend for past years + predicted year
years = list(range(prediction_year - 5, prediction_year + 1))  # Past 5 years + predicted year
capacity_trends = [np.random.uniform(300, 800) for _ in range(5)]  # Simulated past data
capacity_trends.append(predicted_capacity)  # Append predicted value

# Line chart for capacity trends
fig = px.line(
    x=years,
    y=capacity_trends,
    labels={'x': 'Year', 'y': 'Hospital Capacity'},
    title="Hospital Capacity Trend",
    markers=True
)
st.plotly_chart(fig)

# Bar chart for capacity trends
st.subheader("Capacity by Year")
fig_bar = px.bar(
    x=years,
    y=capacity_trends,
    labels={'x': 'Year', 'y': 'Hospital Capacity'},
    title="Hospital Capacity by Year"
)
st.plotly_chart(fig_bar)

# Sidebar instructions
st.sidebar.markdown("""
---
**Instructions:**
1. Adjust the input features using the sliders.
2. Enter the year you want to predict hospital capacity for.
3. View the predicted results and trends through the visualizations.
""")
