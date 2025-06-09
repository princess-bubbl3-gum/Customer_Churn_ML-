#from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# outside function to label age
def label_age(age):
    if age <= 24:
        return 'Young'
    elif age > 24 and age <= 44:
        return 'Adult'
    elif age > 44 and age <= 64:
        return 'Senior'
    elif age > 64:
        return 'Elder'
    else:
        return np.nan

#st.write("Current directory:", os.getcwd())

scaler = joblib.load(os.path.join(os.path.dirname(__file__), "scaler.pkl"))
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))

#scaler = joblib.load("scaler.pkl")
#model = joblib.load("model.pkl")

# 🎨 App Title and Info
st.set_page_config(page_title="Churn Predictor", layout="centered", page_icon="📉")
st.title("📉 Customer Churn Prediction")
st.markdown("Use this app to predict whether a customer is likely to churn.")
st.divider()

# 🧾 Sidebar Inputs
st.sidebar.title("📋 Customer Info")
name = st.sidebar.text_input("👤 Name", placeholder="e.g. Jane Doe")
age = st.sidebar.number_input("🎂 Age", min_value=10, max_value=100, value=30)
gender = st.sidebar.radio("⚧️ Gender", ["Male", "Female"])
tenure = st.sidebar.slider("📅 Tenure (months)", 0, 72, 12)
monthlycharge = st.sidebar.number_input("💸 Monthly Charges", 0.0, 150.0, 50.0)
totalcharge = st.sidebar.number_input("💰 Total Charges", 0.0, 100000.0, 1000.0)
contract_type = st.sidebar.selectbox("📃 Contract Type", ["Month-to-month", "One year", "Two year"])
internetService = st.sidebar.selectbox("🌐 Internet Service", ["DSL", "Fiber Optic", "No Service"])
techsupport = st.sidebar.radio("🛠️ Tech Support", ["Yes", "No"])



st.divider()

# 🧮 Predict Button
predict_button = st.sidebar.button("🔮 Predict")


if predict_button:
    # Create a dict with the input features exactly like your training data columns
    # Example columns - replace with your actual encoded feature names
    feature_dict = {
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthlycharge,
        "TotalCharges": totalcharge,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,          # if you kept both
        "ContractType_Month-to-Month": 1 if contract_type == "Month-to-month" else 0,
        "ContractType_One-Year": 1 if contract_type == "One-Year" else 0,
        "ContractType_Two-Year": 1 if contract_type == "Two-Year" else 0,
        "InternetService_DSL": 1 if internetService == "DSL" else 0,
        "InternetService_Fiber Optic": 1 if internetService == "Fiber Optic" else 0,
        "InternetService_No Service": 1 if internetService == "No Service" else 0,          # if you kept both
        "TechSupport_No": 1 if techsupport == "No" else 0,     # if you kept both
        "TechSupport_Yes": 1 if techsupport == "Yes" else 0,
        "AgeGroup_Adult": 1 if label_age(age) == "Adult" else 0,
        "AgeGroup_Elder": 1 if label_age(age) == "Elder" else 0,
        "AgeGroup_Senior": 1 if label_age(age) == "Senior" else 0,
        "AgeGroup_Young": 1 if label_age(age) == "Young" else 0 
    }

    # Create DataFrame with one row, columns must match model training data exactly
    input_df = pd.DataFrame([feature_dict])

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Interpret prediction
    result = "Yes" if prediction == 1 else "No"


    st.divider()
    st.subheader("📢 Prediction Result")
    if result == "Yes":
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **not likely to churn**.")

    st.markdown(f"**Customer Name:** {name}")
else:
    st.info("👈 Enter all inputs on the left and click **Predict** to see results.")
    