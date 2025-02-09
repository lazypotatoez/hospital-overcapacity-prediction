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
def predict_overcapacity(model, data):
    # Ensure input matches the model's required shape
    # Replace this preprocessing based on your model
    X = data.values.reshape(1, -1)
    predictions = model.predict(X)
    return predictions

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

st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type=["csv"])

if uploaded_file:
    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    # Display uploaded data
    st.subheader("Uploaded Data")
    st.dataframe(data)

    # Allow user to select the columns for prediction
    st.sidebar.subheader("Select Features for Prediction")
    features = st.sidebar.multiselect("Select columns to use for prediction", data.columns)
    
    if len(features) > 0:
        st.write("Using features:", features)
        selected_data = data[features]

        # Prediction
        st.subheader("Prediction Results")
        model = load_prediction_model()
        predictions = predict_overcapacity(model, selected_data.iloc[-1])
        
        # Display prediction
        st.write(f"Predicted Overcapacity Probability: {predictions[0][0]:.2f}")
        
        # Show recommendations
        st.subheader("Resource Optimization Recommendations")
        recommendations = generate_recommendations(predictions)
        st.write(recommendations)

        # Visualization
        st.subheader("Trend Analysis")
        fig = px.line(data, x=data.index, y=features, title="Trend of Selected Features")
        st.plotly_chart(fig)

else:
    st.sidebar.warning("Please upload a CSV file to proceed.")
    st.write("Upload historical data in CSV format to start predictions.")

st.sidebar.markdown("""
---
**Instructions:**
1. Upload a CSV containing historical hospital data (admissions, bed availability, outpatient visits, etc.).
2. Select the features to use for prediction.
3. View predictions and optimization strategies.
""")
