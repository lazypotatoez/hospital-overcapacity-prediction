import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("best_deep_model.h5")

st.title("Hospital Overcapacity Prediction App")
st.write("Predict hospital overcapacity using historical trends.")

# Input features
st.write("Enter the hospital data for prediction:")
features = st.text_input("Enter data as comma-separated values: Admissions, Beds, Outpatient visits, Expenditure (e.g., 500, 200, 300, 50):")


if st.button("Predict"):
    try:
        # Convert the input into a numpy array
        data = np.array([list(map(float, features.split(",")))])
        prediction = model.predict(data)[0][0]
        st.success(f"Predicted Overcapacity: {prediction}")
    except:
        st.error("Invalid input. Please enter valid numerical values.")
