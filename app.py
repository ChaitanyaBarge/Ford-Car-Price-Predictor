import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Ford Car Price Predictor", page_icon="🚗", layout="centered"
)

st.title("🚗 Ford Car Price Prediction")
st.write("Enter the car details below to predict its price.")


# ----------------------------
# Cache & Load Saved Files
# ----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("LR_ford_car.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns


try:
    model, scaler, columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ----------------------------
# User Inputs
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", 2000, 2026, 2018)
    mileage = st.number_input("Mileage", 0, 300000, 50000)
    tax = st.number_input("Tax", 0, 1000, 150)
    engineSize = st.number_input("Engine Size (L)", 0.8, 6.0, 1.5)

with col2:
    mpg = st.number_input("MPG", 0.0, 100.0, 55.0)
    transmission = st.selectbox(
        "Transmission", ["Manual", "Automatic", "Semi-Auto"]
    )
    fuelType = st.selectbox(
        "Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
    )
    model_name = st.selectbox(
        "Model",
        [
            "Fiesta",
            "Focus",
            "Kuga",
            "EcoSport",
            "Mondeo",
            "Ka",
            "B-Max",
            "C-Max",
            "S-Max",
            "Galaxy",
            "Puma",
            "Mustang",
        ],
    )

# ----------------------------
# Prediction Logic
# ----------------------------
if st.button("Predict Price", use_container_width=True):
    try:
        # Create DataFrame for Numerical Inputs
        input_data = pd.DataFrame(
            {
                "year": [year],
                "mileage": [mileage],
                "tax": [tax],
                "mpg": [mpg],
                "engineSize": [engineSize],
            }
        )

        # Scale Numerical Features
        input_data[["year", "mileage", "tax", "mpg", "engineSize"]] = (
            scaler.transform(
                input_data[["year", "mileage", "tax", "mpg", "engineSize"]]
            )
        )

        # One-Hot Encoding Alignment
        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0

        t_col = "transmission_" + transmission
        if t_col in input_data.columns:
            input_data[t_col] = 1

        f_col = "fuelType_" + fuelType
        if f_col in input_data.columns:
            input_data[f_col] = 1

        m_col = "model_" + model_name
        if m_col in input_data.columns:
            input_data[m_col] = 1

        # Reorder columns to match training set
        input_data = input_data[columns]

        # Prediction
        prediction = model.predict(input_data)

        st.success(f"### Estimated Car Price: £{prediction[0]:,.2f}")

    except Exception as e:
        st.error("Prediction Failed!")
        st.write(e)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Created using Streamlit + Scikit-Learn")
# TO THIS:
try:
    model = joblib.load("LR_ford_car.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except Exception as e:
    st.error(f"Detailed Error: {e}")
    st.stop()
    import os
from pathlib import Path
import joblib
import streamlit as st

# Find current directory of app.py
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "LR_ford_car.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()