import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load pre-trained model (replace 'best_deep_model.h5' with your model file)
@st.cache_resource
def load_prediction_model():
    model = load_model('best_deep_model.h5')
    return model

# Prediction function
def predict_overcapacity(model, inputs):
    try:
        # Reshape the input and ensure it's a float
        X = np.array(inputs, dtype=float).reshape(1, -1)

        # Uncomment if scaling is needed (e.g., MinMaxScaler during training)
        # scaler = load_scaler()  # Load the scaler used during training
        # X = scaler.transform(X)

        # Make prediction
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Resource optimization recommendations
def generate_recommendations(predictions):
    if predictions[0][0] > 0.8:  # Example threshold for overcapacity
        return "Increase bed capacity, hire additional staff, and allocate more funds."
    elif predictions[0][0] > 0.5:
        return "Monitor closely and consider minor resource adjustments."
    else:
        return "Current resources are sufficient for predicted demand."

# Streamlit UI
st.title("Hospital Overcapacity Prediction & Resource Optimization")
st.sidebar.header("Input Features")

# Slider inputs for features
admissions = st.sidebar.slider("Monthly Hospital Admissions", 0, 1000, 310)
bed_availability = st.sidebar.slider("Bed Availability (%)", 0, 100, 45)
outpatient_visits = st.sidebar.slider("Monthly Outpatient Visits", 0, 5000, 2500)
health_expenditure = st.sidebar.slider("Govt. Health Expenditure ($M)", 0, 500, 240)

# Combine inputs into a list
inputs = [admissions, bed_availability, outpatient_visits, health_expenditure]

# Load model
model = load_prediction_model()

# Prediction
st.subheader("Prediction Results")
predictions = predict_overcapacity(model, inputs)
if predictions is not None:
    st.write(f"Predicted Overcapacity Probability: {predictions[0][0]:.2f}")

    # Show recommendations
    st.subheader("Resource Optimization Recommendations")
    recommendations = generate_recommendations(predictions)
    st.write(recommendations)

    # Visualization (optional)
    st.subheader("Input Data Visualization")
    fig, ax = plt.subplots()
    feature_names = ["Admissions", "Bed Availability", "Outpatient Visits", "Health Expenditure"]
    ax.bar(feature_names, inputs, color=['blue', 'orange', 'green', 'red'])
    ax.set_title("Input Features")
    ax.set_ylabel("Values")
    st.pyplot(fig)
else:
    st.write("Prediction could not be completed. Check your input values and try again.")

# Sidebar instructions
st.sidebar.markdown("""
---
**Instructions:**
1. Adjust the sliders to input historical data or estimates.
2. View predictions and optimization strategies.
""")