import streamlit as st
import numpy as np
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

# Default features (40 features expected by the model)
default_features = [0] * 40  # Replace with actual values used during training

if prediction_year:
    model = load_prediction_model()
    try:
        predicted_capacity = predict_capacity(model, default_features)

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

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
