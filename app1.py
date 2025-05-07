import streamlit as st
import numpy as np
import joblib
# Normalize age and fare using the same scaler used during training
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = joblib.load('logistic_regression_model.pkl')

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🛳️ Titanic Survival Prediction")
st.markdown("Fill out the form to predict whether a passenger would survive.")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0.42, 80.0, 25.0)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare Price", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.radio("Port of Embarkation", ["C","S", "Q"])  # Note: 'C' was dropped

# Use fixed range values from training data (manually set from earlier)
age_min, age_max = 0.42, 80.0
fare_min, fare_max = 0.0, 512.3292

norm_age = (age - age_min) / (age_max - age_min)
norm_fare = (fare - fare_min) / (fare_max - fare_min)

# Encode sex
sex_encoded = 0 if sex == "Male" else 1

# Encode embarked (we dropped 'C', so only 'Q' and 'S' used)
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Prepare the final input (must match training columns order)
input_data = np.array([[pclass, sex_encoded, norm_age, sibsp, parch, norm_fare, embarked_Q, embarked_S]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎉 The passenger **would survive**!")
    else:
        st.error("💀 The passenger **would not survive**.")
