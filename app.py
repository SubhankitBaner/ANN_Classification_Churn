import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and preprocessing objects
model = load_model('notebook/model.h5', compile=False)
onehot_encoder_geo = joblib.load('encoders/encodersonehot_geo.pkl')
label_encoder_gender = joblib.load('encoders/label_encoder_gender.pkl')
scaler = joblib.load('encoders/scaler.pkl')

# Sidebar
with st.sidebar:
    st.header("ℹ️ About this Project")
    st.write("""
    This app predicts **Customer Churn** using a trained Artificial Neural Network (ANN).  
    - **Churn** means when a customer is likely to leave the bank.  
    - The dataset used is based on European banks (Germany, France, Spain).  
    - Therefore, money-related values are shown in **Euros (€)**.  

    ⚠️ Disclaimer: This is an **educational project** and should not be used for real financial decisions.
    """)


# Main Title
st.title("💳 Customer Churn Prediction App")
st.write("""
This tool predicts the probability of a customer leaving (churning).  
Fill in the customer details below and click **Predict**.
""")

st.markdown("---")

# User Input Section
st.header("📊 Enter Customer Details")

geography = st.selectbox(
    '🌍 Geography (Country/Region of the customer)',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    '👤 Gender',
    label_encoder_gender.classes_
)

age = st.slider(
    '🎂 Age of the customer',
    18, 92, help="Customer's age (between 18 and 92)"
)

credit_score = st.number_input(
    '💳 Credit Score',
    min_value=300, max_value=900, step=1,
    help="A higher score means better creditworthiness"
)

balance = st.number_input(
    '🏦 Account Balance (€ EUR)',
    min_value=0.0, step=100.0,
    help="Customer's bank balance in Euros"
)

estimated_salary = st.number_input(
    '💼 Estimated Salary (€ EUR)',
    min_value=0.0, step=100.0,
    help="Customer's estimated yearly salary in Euros"
)

tenure = st.slider(
    '📅 Tenure (Years with bank)',
    0, 10,
    help="How long the customer has been with the bank"
)

num_of_products = st.slider(
    '🛒 Number of Products',
    1, 4,
    help="Number of bank products the customer is using (credit card, loan, etc.)"
)

has_cr_card = st.selectbox(
    '💳 Has Credit Card?',
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

is_active_member = st.selectbox(
    '✅ Is Active Member?',
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Predict button
if st.button("🔮 Predict"):
    # Prepare input data (without Geography for now)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography separately
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Concatenate encoded geography with other features
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict Churn
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    # Display results
    st.markdown("### 📌 Prediction Result")
    st.write(f"**Churn Probability:** `{prediction_prob:.2f}`")

    if prediction_prob > 0.5:
        st.error("🚨 Customer is **likely to churn** (leave the bank).")
    else:
        st.success("✅ Customer is **not likely to churn**.")
