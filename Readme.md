# 🏦 Bank Customer Churn Prediction System

An end-to-end Machine Learning web application designed to predict customer churn for a banking institution using an **Artificial Neural Network (ANN)**. Built with **TensorFlow / Keras**, **Scikit-Learn**, and deployed interactively via **Streamlit** with real-time risk visualizer gauges using **Plotly**.

---

## 📌 Features

- **Deep Learning Model**: Uses a multi-layer Neural Network (ANN) with ReLU and Sigmoid activations.
- **Class Imbalance Management**: Balanced weight distribution applied during model training to improve minority class recall.
- **Dynamic Input Alignment**: Built-in feature pipeline (`model_features.pkl`) to prevent categorical dummy variable mismatch during single-row online predictions.
- **Interactive UI**: Clean, multi-column dashboard with real-time risk gauges and dynamic probability classification.

---

## 🏗️ Project Architecture

```text
CUSTOMERCHURN/
│
├── app.py                  # Main Streamlit web application
├── train.py                # Model training and artifact export script
├── cutomer_churn_ann.ipynb # Exploratory Data Analysis & ANN development
├── Churn_Modelling.csv     # Raw customer dataset
│
├── model_features.pkl      # Saved feature name list for input schema validation
├── scaler.pkl              # Fitted StandardScaler object
├── ann_model.h5            # Saved Keras ANN model weights & structure
│
├── requirements.txt        # Python dependency list
└── README.md               # Project documentation

## ⚡ Technical Stack

- **Frontend / Dashboard**: Streamlit, Plotly
- **Machine Learning / Deep Learning**: TensorFlow, Keras, Scikit-Learn
- **Data Manipulation**: Pandas, NumPy
- **Serialization**: Joblib

---

## 📊 Model Pipeline Overview

1. **Preprocessing**: Irrelevant identifiers (`RowNumber`, `CustomerId`, `Surname`) are dropped.
2. **Encoding**: Categorical features (`Geography`, `Gender`) are dummy-encoded using One-Hot Encoding.
3. **Scaling**: Features are normalized using `StandardScaler` to balance numeric feature ranges.
4. **Handling Imbalance**: Model parameters adjust for class imbalance via `compute_class_weight`.
5. **ANN Architecture**:
   - **Input Layer**: Matches dynamic input shape (10 features)
   - **Hidden Layer 1**: 16 Neurons + `ReLU`
   - **Hidden Layer 2**: 8 Neurons + `ReLU`
   - **Output Layer**: 1 Neuron + `Sigmoid` (Binary Classification)

---

## 🌐 Deployment

This application is ready for continuous deployment on **Streamlit Community Cloud**:
https://bank-churn-app-rrnmh75kcnndwuarvwft7k.streamlit.app/