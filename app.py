import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np

# Load the trained Random Forest model
#with open('./data/rf_model.pkl', 'rb') as file:
#    rf_model = pickle.load(file)
# Load the trained model
rf_model = joblib.load('rf_model.pkl')

# Define the feature columns - must match the columns used in the training script
feature_cols = [
    'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption',
    'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita',
    'Schooling', 'Economy_status_Developed', 'Economy_status_Developing'
]

# Streamlit UI to get user inputs
st.title("Life Expectancy Prediction")

# Numerical inputs for each feature
Infant_deaths = st.number_input("Infant deaths", min_value=0, max_value=1000, step=0.5, format='%.2f', value=0.0)
Under_five_deaths = st.number_input("Under five deaths", min_value=0, max_value=1000, step=0.5, format='%.2f', value=0.0)
Adult_mortality = st.number_input("Adult mortality", min_value=0, max_value=1000, step=0.5, format='%.2f', value=0.0)
Alcohol_consumption = st.number_input("Alcohol consumption", min_value=0.0, max_value=100.0, step=1, format='%.2f', value=0.0)
Hepatitis_B = st.number_input("Hepatitis B (%)", min_value=0.0, max_value=100.0, step=1, format='%.2f', value=100.0)
Measles = st.number_input("Measles cases", min_value=0, max_value=100.0, step=1, format='%.2f', value=100.0)
BMI = st.number_input("BMI", min_value=0.0, max_value=40.0, step=1, format='%.2f', value=25.0)
Polio = st.number_input("Polio (%)", min_value=0.0, max_value=100.0, step=1, format='%.2f', value=100.0)
Diphtheria = st.number_input("Diphtheria (%)", min_value=0.0, max_value=100.0, step=1, format='%.2f', value=100.0)
Incidents_HIV = st.number_input("Incidents of HIV", min_value=0, max_value=100.0, step=1, format='%.2f', value=0.0)
GDP_per_capita = st.number_input("GDP per capita", min_value=0.0, max_value=150000.0, step=1, format='%.2f', value=0.0)
Schooling = st.number_input("Schooling (years)", min_value=0.0, max_value=20.0, step=1, format='%.2f', value=0.0)

# Dropdown for Economy Status
economy_status = st.selectbox("Economy Status", options=["Developed", "Developing"])

# Set the economy status features based on the dropdown selection
if economy_status == "Developed":
    Economy_status_Developed = 1
    Economy_status_Developing = 0
else:
    Economy_status_Developed = 0
    Economy_status_Developing = 1

# Button to make the prediction
if st.button("Predict Life Expectancy"):
    # Convert the inputs to a DataFrame
    input_data = pd.DataFrame([[
        Infant_deaths, Under_five_deaths, Adult_mortality, Alcohol_consumption,
        Hepatitis_B, Measles, BMI, Polio, Diphtheria, Incidents_HIV, GDP_per_capita,
        Schooling, Economy_status_Developed, Economy_status_Developing
    ]], columns=feature_cols)
    
    # Make the prediction
    prediction = rf_model.predict(input_data)[0]

    # Display the predicted value
    st.success(f"Predicted Life Expectancy: {prediction:.2f} years")
    
