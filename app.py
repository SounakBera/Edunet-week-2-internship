import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(page_title="EV Price Predictor", page_icon="⚡", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found. Please run 'train_model.py' first.")
        return None

model = load_model()

# Header
st.image("https://cdn.pixabay.com/photo/2022/01/25/19/12/electric-car-6968348_1280.jpg", use_container_width=True)
st.title("⚡ EV Price Predictor")
st.markdown("### Adjust the specifications to estimate the car's value.")

# User Input Form
col1, col2 = st.columns(2)

with col1:
    battery = st.number_input("🔋 Battery Capacity (kWh)", min_value=20.0, max_value=150.0, value=75.0)
    accel = st.number_input("🚀 0-100 km/h (sec)", min_value=2.0, max_value=20.0, value=6.0)
    seats = st.slider("🪑 Number of Seats", 2, 8, 5)

with col2:
    top_speed = st.number_input("🏎️ Top Speed (km/h)", min_value=100, max_value=400, value=200)
    range_km = st.number_input("🛣️ Range (km)", min_value=100, max_value=800, value=400)
    efficiency = st.number_input("⚡ Efficiency (Wh/km)", min_value=100, max_value=300, value=180)

# Prediction Logic
if st.button("🔮 Predict Price"):
    if model:
        # Create a DataFrame matching the features used in training EXACTLY
        input_data = pd.DataFrame({
            'Battery': [battery],
            '0-100': [accel],
            'Top_Speed': [top_speed],
            'Range': [range_km],
            'Efficiency': [efficiency],
            'Number_of_seats': [seats]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.success(f"💰 Estimated Price: **${prediction:,.2f}**")
    else:
        st.error("Model could not be loaded.")
