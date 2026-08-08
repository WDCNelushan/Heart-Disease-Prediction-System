# Making the UI for the model using Streamlit
import sys
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:
    sys.exit("Please run this app using: streamlit run app.py")

# Ensure scikit-learn is available for unpickling the saved model
try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn', 'scipy', 'joblib'])
    import sklearn  # noqa: F401

# Load the saved model and scaler
with open('heart_disease_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

# Define the columns
categorical_columns = ['Gender', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseEngage', 'ST_Slope', 'MajorVessels', 'Thalassemia']
numerical_columns = ['Age', 'RestingBp', 'Cholesterol', 'MaxHR', 'ST_Depression']

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("Heart Disease Prediction System")

st.write("This app predicts the likelihood of heart disease based on user input.")

st.markdown("--")

# Collect input data

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=20, max_value=120, value=30)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", options=[0,1,2,3])
    RestingBp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    Cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)

with col2:
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0,1])
    RestingECG = st.selectbox("Resting ECG", options=[0,1,2])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    ExerciseEngage = st.selectbox("Exercise Induced Angina", options=[0,1])
    ST_Depression = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0, step = 0.1)
    ST_Slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0,1,2])
    MajorVessels = st.selectbox("Number of Major Vessels (0-4) Colored by Fluoroscopy", options=[0,1,2,3,4])
    Thalassemia = st.selectbox("Thalassemia (1-3)", options=[1,2,3])

Gender = 1 if gender == "Male" else 0

# Create data frame
input_dict = {
    'Age': age,
    'Gender': Gender,
    'ChestPainType': ChestPainType,
    'RestingBp': RestingBp,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'RestingECG': RestingECG,
    'MaxHR': MaxHR,
    'ExerciseEngage': ExerciseEngage,
    'ST_Depression': ST_Depression,
    'ST_Slope': ST_Slope,
    'MajorVessels': MajorVessels,
    'Thalassemia': Thalassemia
}

input_df = pd.DataFrame([input_dict])

input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

expected_encoded = model.feature_names_in_

input_encoded = input_encoded.reindex(columns=expected_encoded, fill_value=0)

# Scale the input data
input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])  

# Prediction button

if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.error("High risk of heart disease detected! Please consult a healthcare professional for further evaluation.")
    else:
        st.success("No sign of heart disease detected. However, it's always good to maintain a healthy lifestyle and regular check-ups.")

st.caption("Disclaimer: This prediction is based on a machine learning model and should not be considered a medical diagnosis. Always consult with a healthcare professional for medical advice.")

st.caption("Developed by Chamika Nelushan | @2026 | Machine Learning Project")