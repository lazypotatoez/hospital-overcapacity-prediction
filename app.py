import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("best_deep_model.h5")

st.title("Hospital Overcapacity Prediction App")
st.write("Predict hospital overcapacity using historical trends.")

# Input features
st.write("Enter the features for the last 12 months:")
features = st.text_input("Comma-separated feature values (e.g., 100, 200, 300, ...):")

if st.button("Predict"):
    try:
        # Convert the input into a numpy array
        data = np.array([list(map(float, features.split(",")))])
        prediction = model.predict(data)[0][0]
        st.success(f"Predicted Overcapacity: {prediction}")
    except:
        st.error("Invalid input. Please enter valid numerical values.")
