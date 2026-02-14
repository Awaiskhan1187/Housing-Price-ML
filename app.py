import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🏡 Housing Price ML Predictor")

st.write("""
Enter the features below to predict the house price
""")

# --- Load pre-trained model ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# --- User Inputs ---
st.sidebar.header("Input Features")

def user_input_features():
    # Example features (adjust to match your model)
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
    sqft_living = st.sidebar.number_input("Sqft Living", value=1000)
    sqft_lot = st.sidebar.number_input("Sqft Lot", value=5000)
    floors = st.sidebar.selectbox("Floors", [1, 2, 3])
    waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
    view = st.sidebar.slider("View (0–4)", 0, 4, 0)
    grade = st.sidebar.slider("Grade (1–13)", 1, 13, 7)
    yr_built = st.sidebar.number_input("Year Built", value=1990)

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'grade': grade,
        'yr_built': yr_built,
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.write("### Your Input Features")
st.dataframe(input_df)

# --- Predict ---
prediction = model.predict(input_df)
st.write("## 🏷️ Predicted House Price")
st.success(f"${prediction[0]:,.2f}")
