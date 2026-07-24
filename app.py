import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load the saved model and preprocessors
@st.cache_resource
def load_assets():
    ann_model = tf.keras.models.load_model('churn_ann_model.keras')
    scaler = joblib.load('scaler.joblib')
    transformer = joblib.load('column_transformer.joblib')
    return ann_model, scaler, transformer

model, scaler, ct = load_assets()

st.title("🏦 Bank Customer Churn Predictor")
st.write("Enter customer details below to predict their churn probability.")

# Input fields
credit_score = st.number_input("Credit Score", 300, 850, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 38)
tenure = st.slider("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Balance ($)", 0.0, 250000.0, 60000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0)

if st.button("Predict Churn"):
    # Preprocess inputs
    gender_encoded = 1 if gender == "Male" else 0
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_encoded = 1 if is_active_member == "Yes" else 0

    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_encoded],
        'IsActiveMember': [is_active_encoded],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography and scale features
    encoded_input = ct.transform(input_df)
    scaled_input = scaler.transform(encoded_input)

    # Make prediction
    churn_prob = model.predict(scaled_input)[0][0]

    st.subheader("Results")
    st.write(f"**Churn Probability:** `{churn_prob * 100:.2f}%`")

    # Using our custom threshold (~0.35) for churn risk
    if churn_prob > 0.35:
        st.error("⚠️ High Risk of Churn! Consider offering retention incentives.")
    else:
        st.success("✅ Low Risk of Churn. Customer is likely to stay.")