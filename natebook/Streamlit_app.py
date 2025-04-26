import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Credit Risk Prediction App", layout="centered")

# Title
st.title("💳 Credit Risk Prediction App")
st.markdown("This app predicts whether a loan applicant is a **Good** or **Bad** credit risk using the best performing machine learning model.")

# Load model and expected columns
model = joblib.load("models/best_model.pkl")
expected_columns = joblib.load("models/expected_columns.pkl")
scaler = joblib.load("models/scaler.pkl")

# User Input
st.sidebar.header("Input Applicant Details")

age = st.sidebar.slider("Age", 18, 75, 30)
credit_amount = st.sidebar.number_input("Credit Amount", min_value=100, max_value=20000, value=5101)
duration = st.sidebar.slider("Loan Duration (months)", 6, 72, 24)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
housing = st.sidebar.selectbox("Housing", ["own", "free", "rent"])
purpose = st.sidebar.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])
saving_acc = st.sidebar.selectbox("Saving Accounts", ["no_info", "little", "moderate", "rich", "quite rich"])
checking_acc = st.sidebar.selectbox("Checking Account", ["no_info", "little", "moderate", "rich"])

# Create a DataFrame from user input
user_input = pd.DataFrame({
    "Age": [age],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Sex": [sex],
    "Housing": [housing],
    "Purpose": [purpose],
    "Saving accounts": [saving_acc],
    "Checking account": [checking_acc]
})

# Preprocessing: same as training
def preprocess_input(input_df):
    # Fill missing structure (shouldn't happen in UI)
    input_df["Saving accounts"].fillna("no_info", inplace=True)
    input_df["Checking account"].fillna("no_info", inplace=True)

    # One-hot encode
    df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Add missing columns & drop extras
    for col in expected_columns:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]  # Ensure correct order

    # Scale numeric columns
    numeric_features = ['Age', 'Credit amount', 'Duration']
    df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])

    return df_encoded

# Preprocess user input
input_prepared = preprocess_input(user_input)

# Prediction
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_prepared)[0]
    prediction_label = "Good Credit Risk" if prediction == 1 else "Bad Credit Risk"

    st.markdown("---")
    st.subheader("🔍 Prediction Result:")
    st.success(f"The applicant is likely a **{prediction_label}**.")

    st.markdown("✔️ This prediction is made using the best classifier selected from KMeans, DBSCAN, Random Forest, SVM, and Logistic Regression.")














