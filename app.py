import streamlit as st
import pickle
import numpy as np

# load model
with open("nlp_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("NLP Prediction App")

user_input = st.text_area("Enter Text")

if st.button("Predict"):
    try:
        # reshape input
        input_data = np.array([user_input]).reshape(1, -1)

        prediction = model.predict(input_data)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
