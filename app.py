import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Streamlit App Title
st.title("Outpatient Attendance Prediction")

# Sidebar for Inputs
st.sidebar.header("Historical Attendance Data")
sequence_length = st.sidebar.radio("Select Sequence Length:", [1, 12], index=1)

if sequence_length == 12:
    st.sidebar.write("Enter outpatient attendance for the last 12 months:")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    attendance_data = []
    for month in months:
        value = st.sidebar.number_input(f"{month}", min_value=0, value=0, step=100)
        attendance_data.append(value)
else:
    attendance_data = [
        st.sidebar.number_input("Enter outpatient attendance for the previous month:", min_value=0, value=0, step=100)
    ]

# CSV Upload Option
st.sidebar.write("---")
uploaded_file = st.sidebar.file_uploader("Or Upload CSV File:", type=["csv"])

def predict_future(attendance_data, years=5):
    X = np.arange(len(attendance_data)).reshape(-1, 1)
    y = np.array(attendance_data)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(attendance_data), len(attendance_data) + years).reshape(-1, 1)
    predictions = model.predict(future_X)
    return predictions

# Process Uploaded CSV
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'attendance' in data.columns:
        attendance_data = data['attendance'].tolist()
        st.success("CSV uploaded successfully!")
    else:
        st.error("The CSV must contain a column named 'attendance'.")

# Prediction and Visualization
if st.button("Predict Future Attendance"):
    if len(attendance_data) >= 1:
        future_years = 5
        predictions = predict_future(attendance_data, years=future_years)

        # Display Results
        st.subheader("Prediction Results")
        for i, pred in enumerate(predictions):
            st.write(f"Year {i + 1}: {int(pred):,}")

        # Plotting
        st.subheader("Attendance Trends")
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(attendance_data)), attendance_data, label="Historical Data", marker='o')
        plt.plot(
            np.arange(len(attendance_data), len(attendance_data) + future_years),
            predictions,
            label="Predicted Data",
            marker='o',
        )
        plt.xlabel("Time")
        plt.ylabel("Attendance")
        plt.legend()
        plt.title("Outpatient Attendance Trend")
        st.pyplot(plt)
    else:
        st.error("Please provide at least one data point for prediction.")

# Footer
st.write("---")
st.write("Developed with ❤️ using Streamlit")