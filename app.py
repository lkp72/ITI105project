import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained Random Forest model
with open('./data/rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

print("pickle loaded")

# Title for the app
st.title("Life Expectancy Prediction")

# Sidebar for user input features
st.sidebar.header("Input Features")

# Define a function to collect user input
def user_input_features():
    # Add input fields for each feature you used in the model
    gdp_per_capita = st.sidebar.number_input("GDP per Capita", min_value=0, max_value=100000, value=5000)
    schooling = st.sidebar.slider("Schooling (Years)", min_value=0.0, max_value=20.0, value=10.0)
    bmi = st.sidebar.slider("BMI", min_value=0.0, max_value=50.0, value=25.0)
    # Add more fields for other features used in your model
    
    # Store the inputs in a dictionary
    data = {'GDP_per_capita': gdp_per_capita,
            'Schooling': schooling,
            'BMI': bmi
            # Add more features here
            }
    
    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input data
input_df = user_input_features()
print(input_df)

# Display user input data on the main page
st.subheader("User Input Features")
st.write(input_df)

# Make predictions
if st.button('Predict'):
    prediction = rf_model.predict(input_df)
    st.subheader("Predicted Life Expectancy")
    st.write(prediction[0])
