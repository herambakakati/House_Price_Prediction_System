import os
import joblib
import streamlit as st

MODEL_PATH = "house_price_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()
