import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Load Model & Preprocessing Artifacts
# ==========================================


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    expected_features = joblib.load("model_features.pkl")
    model = tf.keras.models.load_model("ann_model.h5")
    return scaler, expected_features, model


try:
    scaler, expected_features, model = load_artifacts()
except Exception as e:
    st.error(
        f"Error loading model artifacts. Make sure `scaler.pkl`, `model_features.pkl`, and `ann_model.h5` exist in the directory."
    )
    st.stop()

# ==========================================
# 3. Header Section
# ==========================================
st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "Fill in the customer details below to assess their probability of churning."
)
st.markdown("---")

# ==========================================
# 4. Multi-Column Input Form
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics & Basic Info")
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=850, value=650, step=1
    )
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=40, step=1
    )
    tenure = st.number_input(
        "Tenure (Years with Bank)",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
    )

with col2:
    st.subheader("💰 Financial & Account Info")
    balance = st.number_input(
        "Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0
    )
    num_of_products = st.number_input(
        "Number of Products", min_value=1, max_value=4, value=2, step=1
    )
    has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input(
        "Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0
    )

st.markdown("---")

# ==========================================
# 5. Prediction Logic & Results
# ==========================================
if st.button("Predict Churn Risk"):
    # 1. Structure Raw Input DataFrame
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

    # 2. One-hot encode single input row
    encoded_input = pd.get_dummies(raw_input)

    # 3. Align features with training feature set (adds missing dummy columns with 0)
    encoded_input = encoded_input.reindex(
        columns=expected_features, fill_value=0
    )

    # 4. Scale inputs and make prediction
    scaled_input = scaler.transform(encoded_input)
    prediction = model.predict(scaled_input)[0][0]
    churn_prob = float(prediction)

    # Display Metrics Summary
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("📋 Status Summary")
        if churn_prob > 0.5:
            st.error("🚨 **High Churn Risk Detected**")
            st.write(
                "This customer shows account activity patterns highly aligned with churn risk. Consider offering proactive retention incentives."
            )
        else:
            st.success("✅ **Low Churn Risk**")
            st.write(
                "This customer exhibits strong retention signals and is likely to remain with the bank."
            )

        st.metric(
            label="Estimated Churn Probability",
            value=f"{churn_prob:.1%}",
            delta="High Risk" if churn_prob > 0.5 else "Low Risk",
            delta_color="inverse",
        )

    with res_col2:
        # Gauge Chart Visualization
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Risk Meter (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": "#dc3545" if churn_prob > 0.5 else "#198754"
                    },
                    "steps": [
                        {"range": [0, 35], "color": "#d1e7dd"},
                        {"range": [35, 65], "color": "#fff3cd"},
                        {"range": [65, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)