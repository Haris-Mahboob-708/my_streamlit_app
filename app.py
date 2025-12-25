import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the model
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'linear_regression_model.pkl' is in the same folder.")
    st.stop()

# 2. Page Configuration (Tab title)
st.set_page_config(page_title="MILP Predictor", layout="wide")

# 3. Sidebar for Inputs
st.sidebar.header("MILP Configuration")
st.sidebar.write("Adjust parameters:")

# Inputs with sliders for better interactivity
feed = st.sidebar.slider("Feed (kg)", min_value=0.0, max_value=30.0, value=11.0, step=0.5)
time = st.sidebar.slider("Milking Time (min)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
thi = st.sidebar.slider("THI (Temp-Humidity Index)", min_value=30.0, max_value=100.0, value=70.0, step=0.1)

# 4. Main Dashboard Area
st.title("MILP")
st.markdown("### Production Prediction Dashboard")

# Create two columns: Left for Prediction, Right for Chart
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    
    # Calculate prediction
    input_data = np.array([[feed, time, thi]])
    prediction = model.predict(input_data)[0]
    
    # Display big metric
    st.metric(label="Predicted Output", value=f"{prediction:.2f}")
    
    # Optional: logic to show if it's high/low (example logic)
    if prediction > 15:
        st.success("High Production Estimate")
    else:
        st.info("Standard Production Estimate")

with col2:
    st.subheader("Input Analysis")
    # Create a DataFrame for the chart
    chart_data = pd.DataFrame({
        "Parameter": ["Feed (kg)", "Milking Time (min)", "THI"],
        "Value": [feed, time, thi]
    })
    
    # Display the bar chart
    st.bar_chart(chart_data.set_index("Parameter"))

# 5. Raw Data Expandable (Optional)
with st.expander("See Raw Data"):
    st.write(f"**Current Configuration:** Feed={feed}kg, Time={time}min, THI={thi}")
    st.write(f"**Calculated Output:** {prediction}")
