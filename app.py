import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt  # <-- IMPORT MATPLOTLIB

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

# Stop execution if model didn't load
if model is None:
    st.stop()

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


# --- NEW SECTION: EXPLORATORY GRAPH (USING MATPLOTLIB) ---
st.divider()
st.header("📊 Explore Feature Relationships")
st.markdown("See how the price changes when you vary one feature (holding all others constant at the values above).")

# 1. Define the features we can plot (continuous ones)
# Format: "Display Name": {"col": "DataFrame Column", "min": min_val, "max": max_val}
feature_options = {
    "🛣️ Range (km)": {"col": "Range", "min": 100, "max": 800},
    "🔋 Battery Capacity (kWh)": {"col": "Battery", "min": 20.0, "max": 150.0},
    "🚀 0-100 km/h (sec)": {"col": "0-100", "min": 2.0, "max": 20.0},
    "🏎️ Top Speed (km/h)": {"col": "Top_Speed", "min": 100, "max": 400},
    "⚡ Efficiency (Wh/km)": {"col": "Efficiency", "min": 100, "max": 300}
}

# 2. Get user's choice for the plot's x-axis
selected_feature_label = st.selectbox(
    "Select a feature to vary:",
    options=list(feature_options.keys())
)

# 3. Get the "constant" values from the input widgets
base_inputs = {
    'Battery': battery,
    '0-100': accel,
    'Top_Speed': top_speed,
    'Range': range_km,
    'Efficiency': efficiency,
    'Number_of_seats': seats
}

# 4. Generate and display the plot
if selected_feature_label:
    try:
        feature_info = feature_options[selected_feature_label]
        feature_col = feature_info["col"]
        
        # Create an array of 50 values for the x-axis (the feature we're varying)
        plot_x_values = np.linspace(feature_info["min"], feature_info["max"], 50)
        
        # Create the DataFrame for prediction
        plot_df = pd.DataFrame([base_inputs] * 50)
        
        # Overwrite the selected feature's column with the 50 new values
        plot_df[feature_col] = plot_x_values
        
        # Get predictions for all 50 hypothetical cars
        plot_y_predictions = model.predict(plot_df)
        
        # --- MATPLOTLIB PLOTTING ---
        # 1. Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 2. Plot the data
        ax.plot(plot_x_values, plot_y_predictions, color="dodgerblue", linewidth=2)
        
        # 3. Style the plot
        ax.set_title(f"Price vs. {selected_feature_label}", fontsize=16)
        ax.set_xlabel(selected_feature_label, fontsize=12)
        ax.set_ylabel("Estimated Price ($)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Format y-axis as currency
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 4. Display the plot in Streamlit
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"An error occurred while generating the plot: {e}")
