import streamlit as st
import joblib
import numpy as np

# 1. Load the model
# We use the exact filename you provided
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'linear_regression_model.pkl' is in the same folder.")
    st.stop()

# 2. App Title and Description
st.title("Dairy Production Predictor")
st.write("Enter the parameters below to predict the outcome based on your Linear Regression model.")

# 3. Create Input Fields
# Your model expects: ['Feed_kg', 'Milking_Time_min', 'THI']
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    feed = st.number_input("Feed (kg)", min_value=0.0, value=10.0, step=0.1)

with col2:
    time = st.number_input("Milking Time (min)", min_value=0.0, value=5.0, step=0.1)

with col3:
    thi = st.number_input("THI (Temp-Humidity Index)", min_value=0.0, value=50.0, step=0.1)

# 4. Prediction Logic
if st.button("Predict Result"):
    # Create the input array in the EXACT order the model expects
    input_data = np.array([[feed, time, thi]])
    
    try:
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result
        st.success(f"Predicted Output: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")