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
# Helpers: preset examples
# --------------------
def make_zero_example():
    return {col: 0.0 for col in FEATURE_COLUMNS}

def make_fraud_like_example():
    # Exemple "fraude-like" (valeurs typiques extrêmes sur plusieurs composantes PCA)
    ex = make_zero_example()
    ex.update({
        "Time": 40000.0,
        "Amount": 1200.0,
        "V1": -3.5,  "V2": 2.8,  "V3": -4.2, "V4": 2.1,  "V5": -1.9, "V6": -2.0,
        "V7": -3.1,  "V8": 1.2,  "V9": -2.5, "V10": -4.0, "V11": 2.7, "V12": -3.6,
        "V13": 0.5,  "V14": -4.5, "V15": 0.2,  "V16": -2.8, "V17": -3.9, "V18": -1.1,
        "V19": 0.6,  "V20": 1.9,  "V21": 0.8,  "V22": 0.4,  "V23": -0.6, "V24": 0.3,
        "V25": -0.2, "V26": -0.1, "V27": 0.5,  "V28": -0.3,
    })
    return ex

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")

st.title("💳 Détection de fraude bancaire")
st.write("Mini dashboard de démonstration basé sur un modèle XGBoost.")

st.markdown("---")

threshold = st.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=float(DEFAULT_THRESHOLD),
    step=0.01
)

st.markdown("---")

mode = st.radio(
    "Mode de saisie",
    ["Simplifié (Time, Amount)", "Expert (toutes les variables)"],
    horizontal=True
)
st.caption(f"Mode actuel : {mode}")

st.markdown("---")
st.subheader("Simulation d'une transaction")

# --------------------
# Session state: keep values between reruns
# --------------------
if "input_data" not in st.session_state:
    st.session_state.input_data = make_zero_example()

colA, colB = st.columns(2)

with colA:
    if st.button("✅ Charger un exemple normal"):
        st.session_state.input_data = make_zero_example()
        st.session_state.input_data["Time"] = 10000.0
        st.session_state.input_data["Amount"] = 50.0

with colB:
    if st.button("🚨 Charger un exemple de fraude"):
        st.session_state.input_data = make_fraud_like_example()

input_data = dict(st.session_state.input_data)

# --------------------
# Inputs
# --------------------
if mode.startswith("Simplifié"):
    input_data["Time"] = st.number_input("Time", value=float(input_data.get("Time", 0.0)))
    input_data["Amount"] = st.number_input("Amount", value=float(input_data.get("Amount", 0.0)))
else:
    for col in FEATURE_COLUMNS:
        input_data[col] = st.number_input(col, value=float(input_data.get(col, 0.0)))

# Persist
st.session_state.input_data = input_data

# Build DF in correct order
input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

# --------------------
# Prediction
# --------------------
if st.button("Analyser la transaction"):
    proba = model.predict_proba(input_df)[0, 1]
    st.write(f"Probabilité de fraude : **{proba:.4f}**")

    if proba >= threshold:
        st.error("🚨 Transaction suspecte (FRAUDE)")
    else:
        st.success("✅ Transaction normale")
