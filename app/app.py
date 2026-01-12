import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --------------------
# Load model
# --------------------
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_model.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
DEFAULT_THRESHOLD = bundle["threshold"]
FEATURE_COLUMNS = bundle["feature_columns"]

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")

st.title("💳 Détection de fraude bancaire")
st.write("Mini dashboard de démonstration basé sur un modèle XGBoost.")

st.markdown("---")

# Threshold slider
threshold = st.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=float(DEFAULT_THRESHOLD),
    step=0.01
)

st.markdown("---")

# Input form
st.subheader("Simulation d'une transaction")

input_data = {}

for col in FEATURE_COLUMNS:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Analyser la transaction"):
    proba = model.predict_proba(input_df)[0, 1]

    st.write(f"Probabilité de fraude : **{proba:.4f}**")

    if proba >= threshold:
        st.error("🚨 Transaction suspecte (FRAUDE)")
    else:
        st.success("✅ Transaction normale")
