import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the saved model and scalers
model = load_model('cnn_model.h5')

with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

def predict_temperature(date, model, scaler_X, scaler_y, time_step=10):
    # Convert date to day of the year
    date = pd.to_datetime(date)
    day_of_year = date.dayofyear
    
    # Prepare input data
    input_data = np.array([day_of_year] * time_step).reshape(-1, 1)
    input_data = scaler_X.transform(input_data)
    
    # Reshape input data for the model
    input_data = input_data.reshape((1, time_step, 1))
    
    # Make prediction
    y_pred_scaled = model.predict(input_data)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred[0, 0]

# Streamlit app layout
st.title('Temperature Prediction App of Seattle, Washington, USA')

date_input = st.date_input("Select a Date")
predict_button = st.button("Predict Temperature")

if predict_button:
    predicted_temp = predict_temperature(date_input, model, scaler_X, scaler_y, time_step=10)
    st.markdown(f'### Predicted Temperature for {date_input}: **{predicted_temp:.2f}°C**', unsafe_allow_html=True)
