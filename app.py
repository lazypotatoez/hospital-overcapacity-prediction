import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
@st.cache_resource
def load_prediction_model():
    model = load_model('best_deep_model.h5')
    return model

# Prediction function
def predict_overcapacity(model, inputs):
    X = np.array(inputs).reshape(1, -1)  # Reshape inputs for the model
    predictions = model.predict(X)
    return predictions

# Resource optimization recommendations
def generate_recommendations(predictions):
    if predictions[0][0] > 0.8:  # Threshold for overcapacity
        return "Increase bed capacity, hire additional staff, and allocate more funds."
    elif predictions[0][0] > 0.5:
        return "Monitor closely and consider minor resource adjustments."
    else:
        return "Current resources are sufficient for predicted demand."

# Streamlit UI
st.title("Hospital Overcapacity Prediction & Resource Optimization")

st.sidebar.header("Input Features")
st.sidebar.markdown("Provide historical data or adjust sliders to predict future hospital capacity.")

# Example input features (adjust based on your model)
admissions = st.sidebar.slider("Monthly Hospital Admissions", min_value=0, max_value=1000, value=500, step=10)
bed_availability = st.sidebar.slider("Bed Availability (%)", min_value=0, max_value=100, value=70, step=1)
outpatient_visits = st.sidebar.slider("Monthly Outpatient Visits", min_value=0, max_value=5000, value=2000, step=100)
government_health_expenditure = st.sidebar.slider("Govt. Health Expenditure ($M)", min_value=0, max_value=500, value=100, step=10)

# Combine inputs
inputs = [admissions, bed_availability, outpatient_visits, government_health_expenditure]

# Load the model
model = load_prediction_model()

# Prediction
st.subheader("Prediction Results")
predictions = predict_overcapacity(model, inputs)

if predictions is not None:
    st.write(f"Predicted Overcapacity Probability: {predictions[0][0]:.2f}")
    # Display recommendations
    st.subheader("Resource Optimization Recommendations")
    recommendations = generate_recommendations(predictions)
    st.write(recommendations)
else:
    st.error("Error in generating prediction.")

# Instructions
st.sidebar.markdown("""
---
**Instructions:**
1. Adjust the input sliders to reflect your hospital's historical data or expected scenarios.
2. View predictions and recommendations for resource optimization.
""")
