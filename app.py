import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("🏡 Housing Price Predictor")

# Load dataset
data = pd.read_csv("housing.csv")

# Select features (make sure these columns exist in your dataset)
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'grade']]
y = data['price']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.success("Model trained successfully!")

# User input
sqft = st.number_input("Square Footage", 500, 10000, 1000)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
floors = st.slider("Floors", 1, 3, 1)
grade = st.slider("Grade", 1, 13, 7)

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'sqft_living':[sqft],
        'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
        'floors':[floors],
        'grade':[grade]
    })

    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
