import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("car_price_model.pkl")

st.title("🚗 Car Price Prediction App")

# Inputs
year = st.number_input("Year", 2000, 2025)
present_price = st.number_input("Present Price (in lakhs)")
kms_driven = st.number_input("KMs Driven")

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Owners", [0, 1, 2, 3])

# Convert categorical to numeric
fuel_type = 0 if fuel_type == "Petrol" else 1
seller_type = 0 if seller_type == "Dealer" else 1
transmission = 0 if transmission == "Manual" else 1

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]])
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: {round(prediction[0], 2)} Lakhs")