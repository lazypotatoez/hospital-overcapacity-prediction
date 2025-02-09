import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("best_deep_model.h5")

# Set the expected feature count (e.g., 4 features: admissions, beds, visits, expenditure)
expected_feature_count = 4

st.title("Hospital Overcapacity Prediction App")
st.write("Predict hospital overcapacity using historical trends.")

# Input features
st.write("Enter the hospital data for prediction:")
features = st.text_input("Enter data as comma-separated values: Admissions, Beds, Outpatient visits, Expenditure (e.g., 500, 200, 300, 50):")

if st.button("Predict"):
    try:
        # Convert the input into a numpy array
        data = np.array([list(map(float, features.split(",")))])
        
        # Check if input matches the expected feature count
        if data.shape[1] != expected_feature_count:
            st.error(f"Expected {expected_feature_count} features, but got {data.shape[1]}. Please recheck your input.")
        else:
            # Predict overcapacity
            prediction = model.predict(data)[0][0]
            st.success(f"Predicted Overcapacity: {prediction}")
    except ValueError:
        st.error("Invalid input. Please enter valid numerical values separated by commas.")
