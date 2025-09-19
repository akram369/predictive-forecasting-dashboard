import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------------
# Helpers
# ----------------------------
VERSION_DIR = "versions"
CHAMPION_FILE = os.path.join(VERSION_DIR, "champion.json")

if not os.path.exists(VERSION_DIR):
    os.makedirs(VERSION_DIR)


def save_metadata(version, model_name, rmse):
    meta = {
        "version": version,
        "model_name": model_name,
        "rmse": rmse,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    path = os.path.join(VERSION_DIR, version, "metadata.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f)
    return meta


def load_metadata(version):
    path = os.path.join(VERSION_DIR, version, "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def list_versions():
    return [d for d in os.listdir(VERSION_DIR) if os.path.isdir(os.path.join(VERSION_DIR, d))]


def save_champion(version):
    with open(CHAMPION_FILE, "w") as f:
        json.dump({"champion": version}, f)


def load_champion():
    if os.path.exists(CHAMPION_FILE):
        with open(CHAMPION_FILE, "r") as f:
            return json.load(f).get("champion")
    return None

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("AI-based Predictive Demand Forecasting")
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Upload Data", "Train Model", "Compare Versions", "Champion Model"])

# ----------------------------
# Upload Section
# ----------------------------
if menu == "Upload Data":
    file = st.file_uploader("Upload dataset (CSV)", type="csv")
    if file:
        df = pd.read_csv(file)
        st.session_state["data"] = df
        st.write("Preview:", df.head())

# ----------------------------
# Training Section
# ----------------------------
if menu == "Train Model":
    if "data" not in st.session_state:
        st.warning("Please upload data first")
    else:
        model_choice = st.selectbox("Choose Model", ["ARIMA", "XGBoost"])
        df = st.session_state["data"]

        if st.button("Train"):
            y = df.iloc[:, -1].values  # assume last col is target

            if model_choice == "ARIMA":
                model = ARIMA(y, order=(5,1,0))
                model_fit = model.fit()
                preds = model_fit.predict(start=1, end=len(y)-1)
                rmse = np.sqrt(mean_squared_error(y[1:], preds))

            elif model_choice == "XGBoost":
                X = np.arange(len(y)).reshape(-1,1)
                model = XGBRegressor()
                model.fit(X, y)
                preds = model.predict(X)
                rmse = np.sqrt(mean_squared_error(y, preds))

            version = f"{model_choice.lower()}_{int(time.time())}"
            joblib.dump(model, os.path.join(VERSION_DIR, version, "model.pkl"))
            meta = save_metadata(version, model_choice, rmse)

            st.success(f"Trained {model_choice} with RMSE: {rmse:.2f}")
            st.json(meta)

# ----------------------------
# Compare Section
# ----------------------------
if menu == "Compare Versions":
    version_list = list_versions()
    if len(version_list) < 2:
        st.info("Need at least 2 versions to compare")
    else:
        v1 = st.selectbox("Select Version 1", version_list, key="compare_v1")
        v2 = st.selectbox("Select Version 2", version_list, key="compare_v2")

        if v1 and v2 and v1 != v2:
            data = []
            for v in [v1, v2]:
                meta = load_metadata(v)
                if meta:
                    data.append({
                        "Version": v,
                        "Model": meta["model_name"],
                        "RMSE": meta["rmse"]
                    })

            df = pd.DataFrame(data)
            st.write(df)

            plt.figure(figsize=(8,5))
            sns.barplot(x="Version", y="RMSE", hue="Model", data=df)
            plt.title("RMSE Comparison of Selected Versions")
            st.pyplot(plt)

# ----------------------------
# Champion Section
# ----------------------------
if menu == "Champion Model":
    version_list = list_versions()
    champion = load_champion()

    st.write("Current Champion:", champion if champion else "None")

    choice = st.selectbox("Select Champion Version", version_list)
    if st.button("Set as Champion"):
        save_champion(choice)
        st.success(f"Champion updated to {choice}")
