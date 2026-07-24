import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

# Load artifacts
scaler = joblib.load("scaler.pkl")
expected_features = joblib.load("model_features.pkl")
model = tf.keras.models.load_model("ann_model.h5")

st.title("Bank Customer Churn Prediction")

# Streamlit User Input Widgets
credit_score = st.number_input("Credit Score", value=600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", value=40)
tenure = st.number_input("Tenure", value=3)
balance = st.number_input("Balance", value=60000.0)
num_of_products = st.number_input("Number of Products", value=2)
has_cr_card = st.selectbox("Has Credit Card?", [1, 0])
is_active_member = st.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

if st.button("Predict"):
    # 1. Create DataFrame matching original structure before get_dummies
    raw_input = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_of_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    # 2. One-hot encode
    encoded_input = pd.get_dummies(raw_input)

    # 3. Align columns to match exact training feature shape/order (Fills missing dummies with 0)
    encoded_input = encoded_input.reindex(
        columns=expected_features, fill_value=0
    )

    # 4. Scale inputs and predict
    scaled_input = scaler.transform(encoded_input)
    prediction = model.predict(scaled_input)[0][0]

    st.write(f"**Churn Probability:** {prediction:.2%}")
    if prediction > 0.5:
        st.error("Customer is likely to churn!")
    else:
        st.success("Customer is likely to stay.")