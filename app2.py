import streamlit as st
import joblib
import numpy as np

# Load models
logreg_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details to predict whether they would have survived the Titanic disaster.")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    ("Logistic Regression", "Random Forest")
)

# Input fields
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", min_value=0.42, max_value=80.0, value=25.0)
fare = st.slider("Fare", min_value=0.0, max_value=512.0, value=32.0)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=6, value=0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Preprocessing
sex_encoded = 0 if sex == 'male' else 1
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Input vector
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_Q, embarked_S]])

# Normalize Age and Fare
input_data[:, [2, 5]] = scaler.transform(input_data[:, [2, 5]])

# Make prediction
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = logreg_model.predict(input_data)[0]
    else:
        prediction = rf_model.predict(input_data)[0]

    result = "🟩 Survived!" if prediction == 1 else "🟥 Did Not Survive"
    st.subheader(f"Prediction: {result}")
