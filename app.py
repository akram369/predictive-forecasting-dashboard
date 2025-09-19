import os
import io
import json
import zipfile
import joblib
import threading
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Local utilities: model_versioning, utils.model_monitor are expected to exist in project
from model_versioning import save_model_version
from utils.model_monitor import log_model_run

warnings.filterwarnings("ignore")

# ---------------------- CONFIG ----------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_DIR = os.path.join(ROOT_DIR, "model_versions")
CHAMPION_MODEL_PATH = os.path.join(VERSION_DIR, "{version}", "model.pkl")
SCALER_PATH = os.path.join(VERSION_DIR, "{version}", "scaler.pkl")
os.makedirs(VERSION_DIR, exist_ok=True)

st.set_page_config(page_title="📦 Demand Forecasting", layout="wide")
st.title("📈 Predictive Demand Forecasting Dashboard")

# ---------------------- Holiday config ----------------------
default_holidays = [
    "2024-01-01",
    "2024-12-25",
    "2024-11-28",
    "2024-07-04",
]

try:
    with open(os.path.join(ROOT_DIR, "config", "holidays.json"), "r") as f:
        holiday_data = json.load(f)
        holiday_dates = pd.to_datetime(holiday_data if isinstance(holiday_data, list) else default_holidays)
except Exception:
    holiday_dates = pd.to_datetime(default_holidays)

# ---------------------- Helpers ----------------------

def save_version_artifacts(version_folder: str, model_obj, metadata: dict, scaler_obj=None):
    os.makedirs(version_folder, exist_ok=True)
    model_path = os.path.join(version_folder, "model.pkl")
    joblib.dump(model_obj, model_path)
    if scaler_obj is not None:
        joblib.dump(scaler_obj, os.path.join(version_folder, "scaler.pkl"))
    meta_path = os.path.join(version_folder, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def list_versions():
    if not os.path.exists(VERSION_DIR):
        return []
    versions = [d for d in sorted(os.listdir(VERSION_DIR)) if os.path.isdir(os.path.join(VERSION_DIR, d))]
    return versions


def load_metadata(version: str):
    path = os.path.join(VERSION_DIR, version, "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def get_champion_meta():
    best = None
    best_rmse = float("inf")
    for v in list_versions():
        m = load_metadata(v)
        if m and m.get("rmse", float("inf")) < best_rmse:
            best_rmse = m["rmse"]
            best = {"version": v, "model_name": m.get("model_name"), "rmse": m.get("rmse")}
    return best


def get_champion_prediction(input_features: dict, version: str):
    p = os.path.join(VERSION_DIR, version, "model.pkl")
    if not os.path.exists(p):
        return {"error": "Champion model not found."}
    model = joblib.load(p)
    X = pd.DataFrame([input_features])
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        return {"error": f"Model predict error: {e}"}
    return {"forecast": float(pred), "model": m.get("model_name", "Champion"), "status": "success"}

# ---------------------- UI: File Upload ----------------------
uploaded_file = st.file_uploader("Upload CSV or Excel (InvoiceDate & Quantity required)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            df_raw = pd.read_excel(uploaded_file)

        if not set(["InvoiceDate", "Quantity"]).issubset(df_raw.columns):
            st.error("Uploaded file must contain 'InvoiceDate' and 'Quantity' columns.")
            st.stop()

        df = df_raw[["InvoiceDate", "Quantity"]].copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"File Error: {e}")
        st.stop()

    # ---------------------- PREPROCESS ----------------------
    daily_demand = df.set_index("InvoiceDate").resample("D")["Quantity"].sum().fillna(0)
    data = daily_demand.to_frame(name="Quantity")
    data["is_promo"] = (data.index.weekday == 4).astype(int)
    data["is_holiday"] = data.index.isin(holiday_dates).astype(int)

    for lag in range(1, 8):
        data[f"lag_{lag}"] = data["Quantity"].shift(lag)

    data.dropna(inplace=True)

    st.subheader("📊 Daily Demand Overview")
    st.line_chart(daily_demand)

    # ---------------------- MODEL PICK ----------------------
    model_choice = st.selectbox("Select Forecasting Model", ["XGBoost", "ARIMA", "LSTM"])
    fig, ax = plt.subplots(figsize=(10, 4))
    forecast_df = pd.DataFrame()

    # ---------------------- XGBoost ----------------------
    if model_choice == "XGBoost":
        st.subheader("🚀 XGBoost Forecasting")
        X = data.drop(columns=["Quantity"]).copy()
        y = data["Quantity"].copy()

        try:
            model = XGBRegressor()
            model.fit(X, y)
            y_pred = model.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            st.metric("📉 XGBoost RMSE", f"{rmse:.2f}")

            ax.plot(data.index, y, label="Actual")
            ax.plot(data.index, y_pred, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

            # SHAP plot
            st.subheader("🔍 SHAP Summary Plot")
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig_shap)
            plt.clf()

            forecast_df = pd.DataFrame({"Date": data.index, "Actual": y.values, "Predicted": y_pred})

            # Save version
            ts = datetime.utcnow().isoformat()
            version_name = f"xgb_{int(datetime.utcnow().timestamp())}"
            metadata = {"model_name": "XGBoost", "version": version_name, "rmse": rmse, "timestamp": ts}
            version_folder = os.path.join(VERSION_DIR, version_name)
            save_version_artifacts(version_folder, model, metadata)
            log_model_run("XGBoost", "RMSE", rmse)

        except Exception as e:
            st.error(f"⚠️ Error during XGBoost modeling: {e}")

    # ---------------------- ARIMA ----------------------
    elif model_choice == "ARIMA":
        st.subheader("📈 ARIMA Forecasting")
        try:
            train_size = int(len(daily_demand) * 0.8)
            train, test = daily_demand[:train_size], daily_demand[train_size:]
            model = ARIMA(train, order=(5, 1, 2))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            forecast.index = test.index
            rmse = float(np.sqrt(mean_squared_error(test, forecast)))
            st.metric("📉 ARIMA RMSE", f"{rmse:.2f}")

            ax.plot(train.index, train, label="Train")
            ax.plot(test.index, test, label="Test")
            ax.plot(test.index, forecast, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

            forecast_df = pd.DataFrame({"Date": test.index, "Actual": test.values, "Predicted": forecast.values})

            ts = datetime.utcnow().isoformat()
            version_name = f"arima_{int(datetime.utcnow().timestamp())}"
            metadata = {"model_name": "ARIMA", "version": version_name, "rmse": rmse, "timestamp": ts}
            version_folder = os.path.join(VERSION_DIR, version_name)
            save_version_artifacts(version_folder, model_fit, metadata)
            log_model_run("ARIMA", "RMSE", rmse)

        except Exception as e:
            st.error(f"ARIMA modeling error: {e}")

    # ---------------------- LSTM ----------------------
    elif model_choice == "LSTM":
        st.subheader("🤖 LSTM Forecasting")
        try:
            scaler = MinMaxScaler()
            scaled_qty = scaler.fit_transform(data[["Quantity"]])

            def create_sequences(arr, window=30):
                Xs, ys = [], []
                for i in range(len(arr) - window):
                    Xs.append(arr[i : i + window])
                    ys.append(arr[i + window])
                return np.array(Xs), np.array(ys)

            window = 30
            X_seq, y_seq = create_sequences(scaled_qty, window)
            split = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            test_dates = data.index[window + split :]

            model = Sequential()
            model.add(LSTM(64, activation="relu", input_shape=(window, 1)))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

            y_pred = model.predict(X_test)
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_test_inv = scaler.inverse_transform(y_test)

            rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
            st.metric("📉 LSTM RMSE", f"{rmse:.2f}")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(test_dates, y_test_inv, label="Actual")
            ax.plot(test_dates, y_pred_inv, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

            forecast_df = pd.DataFrame({"Date": test_dates, "Actual": y_test_inv.flatten(), "Predicted": y_pred_inv.flatten()})

            ts = datetime.utcnow().isoformat()
            version_name = f"lstm_{int(datetime.utcnow().timestamp())}"
            metadata = {"model_name": "LSTM", "version": version_name, "rmse": rmse, "timestamp": ts}
            version_folder = os.path.join(VERSION_DIR, version_name)
            save_version_artifacts(version_folder, model, metadata, scaler)
            log_model_run("LSTM", "RMSE", rmse)

        except Exception as e:
            st.error(f"LSTM error: {e}")

    # ---------------------- EXPORT ----------------------
    if not forecast_df.empty:
        csv_buf = io.StringIO()
        forecast_df.to_csv(csv_buf, index=False)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("forecast.csv", csv_buf.getvalue())
            z.writestr("forecast_plot.png", img_buf.getvalue())
        st.download_button("📦 Download Forecast ZIP", data=zip_buf.getvalue(), file_name="forecast_bundle.zip", mime="application/zip")

    # ---------------------- SIDEBAR: Version viewer ----------------------
    st.sidebar.title("🧠 Saved Model Versions")
    versions = list_versions()
    if versions:
        selected_version = st.sidebar.selectbox("Select Version Folder", versions)
        meta = load_metadata(selected_version)
        if meta:
            st.sidebar.markdown("#### Model Metadata")
            meta_df = pd.DataFrame(list(meta.items()), columns=["Field", "Value"])
            st.sidebar.table(meta_df)
    else:
        st.sidebar.info("No saved versions yet. Train a model to create versions.")

    # ---------------------- SIDEBAR: Model comparator ----------------------
    st.sidebar.title("📊 Model Comparator")
    versions = list_versions()
    if len(versions) >= 2:
        v1 = st.sidebar.selectbox("Compare Version 1", versions, key="compare_v1")
        v2 = st.sidebar.selectbox("Compare Version 2", versions, key="compare_v2")

        meta1 = load_metadata(v1)
        meta2 = load_metadata(v2)

        if meta1 and meta2:
            st.subheader("📊 Model Comparison Dashboard")
            comparison_data = {
                "Metric": ["Model Type", "Version", "RMSE", "Training Date"],
                f"Model 1 (v{meta1['version']})": [meta1["model_name"], meta1["version"], f"{meta1['rmse']:,.2f}", meta1.get("timestamp")],
                f"Model 2 (v{meta2['version']})": [meta2["model_name"], meta2["version"], f"{meta2['rmse']:,.2f}", meta2.get("timestamp")],
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 📈 RMSE Comparison")
            chart_data = pd.DataFrame({
                "Version": [f"v{meta1['version']}", f"v{meta2['version']}"] ,
                "Model": [meta1["model_name"], meta2["model_name"]],
                "RMSE": [meta1["rmse"], meta2["rmse"]],
            })
            fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
            sns.barplot(data=chart_data, x="Version", y="RMSE", hue="Model", ax=ax_cmp)
            ax_cmp.set_title("RMSE Comparison of Selected Versions")
            plt.xticks(rotation=45)
            st.pyplot(fig_cmp)

            best_model = meta1 if meta1["rmse"] < meta2["rmse"] else meta2
            st.success(f"🏆 Champion: {best_model['model_name']} (v{best_model['version']}) with RMSE {best_model['rmse']:,.2f}")
        else:
            st.warning("⚠️ Metadata missing for one or both selected versions.")
    else:
        st.sidebar.info("Need at least 2 versions for comparison. Train more models.")

    # ---------------------- SIDEBAR: Champion ----------------------
    st.sidebar.title("🏆 Champion Model")
    champion = get_champion_meta()
    if champion:
        st.sidebar.success(f"Champion Model: {champion['model_name']} (v{champion['version']})")
        st.sidebar.info(f"RMSE: {champion['rmse']:.2f}")
    else:
        st.sidebar.warning("No champion model found. Train a model to create a champion.")

    # ---------------------- Champion Forecast ----------------------
    st.subheader("🌐 Real-time Forecast with Champion")
    forecast_days = st.number_input("Forecast how many days ahead?", min_value=1, max_value=30, value=5)

    if st.button("Forecast with Champion"):
        if not champion:
            st.error("No champion model available. Please train a model first.")
        else:
            try:
                recent = daily_demand[-7:].values.tolist()
                today = pd.Timestamp.now().normalize()
                is_promo = 1 if today.weekday() == 4 else 0
                is_holiday = 1 if today.strftime("%Y-%m-%d") in holiday_dates.strftime("%Y-%m-%d").tolist() else 0

                input_data = {"is_promo": is_promo, "is_holiday": is_holiday}
                for j in range(1, 8):
                    input_data[f"lag_{j}"] = int(recent[-j])

                # We only support XGBoost champion for single-step predict here
                p = os.path.join(VERSION_DIR, champion["version"], "model.pkl")
                model = joblib.load(p)
                pred = model.predict(pd.DataFrame([input_data]))[0]
                st.success(f"Champion forecast for {forecast_days} days ahead (next day sample): {pred:.2f}")
            except Exception as e:
                st.error(f"Error during forecasting: {e}")

    # ---------------------- Anomaly detection & Alerts ----------------------
    st.subheader("🚨 Anomaly Detection & Alerting")
    send_email = st.checkbox("Enable Email Alerts", value=False)
    alert_threshold = st.slider("Anomaly Threshold (%)", 5, 100, 20)
    actual_input = st.number_input("Enter Actual Demand (today)", min_value=0.0)

    EMAIL_USER = st.secrets.get("email_user") or os.getenv("EMAIL_USER")
    EMAIL_PASS = st.secrets.get("email_pass") or os.getenv("EMAIL_PASS")

    if st.button("📊 Check for Anomaly"):
        if not champion:
            st.error("No champion model available for anomaly detection")
        else:
            try:
                version_path = os.path.join(VERSION_DIR, champion["version"])
                meta = load_metadata(champion["version"])
                model_type = meta.get("model_name")

                if model_type == "XGBoost":
                    model = joblib.load(os.path.join(version_path, "model.pkl"))
                    recent = daily_demand[-7:].values.tolist()
                    today = pd.Timestamp.now().normalize()
                    is_promo = 1 if today.weekday() == 4 else 0
                    is_holiday = 1 if today.strftime("%Y-%m-%d") in holiday_dates.strftime("%Y-%m-%d").tolist() else 0
                    input_data = {"is_promo": is_promo, "is_holiday": is_holiday}
                    for j in range(1, 8):
                        input_data[f"lag_{j}"] = int(recent[-j])
                    pred = model.predict(pd.DataFrame([input_data]))[0]

                elif model_type == "ARIMA":
                    model = joblib.load(os.path.join(version_path, "model.pkl"))
                    pred = model.forecast(steps=1)[0]

                elif model_type == "LSTM":
                    model = joblib.load(os.path.join(version_path, "model.pkl"))
                    scaler = joblib.load(os.path.join(version_path, "scaler.pkl"))
                    recent = daily_demand[-30:].values.reshape(-1, 1)
                    scaled_recent = scaler.transform(recent)
                    pred = model.predict(scaled_recent.reshape(1, 30, 1))
                    pred = scaler.inverse_transform(pred)[0][0]

                deviation = abs(pred - actual_input) / max(actual_input, 1) * 100
                st.metric("Predicted", round(pred, 2))
                st.metric("Deviation", f"{deviation:.2f}%")

                if deviation > alert_threshold:
                    st.warning(f"⚠️ Anomaly Detected! Deviation: {deviation:.2f}%")
                    if send_email and EMAIL_USER and EMAIL_PASS:
                        st.info("Email alert would be sent here (email sending not implemented in this template)")
                else:
                    st.success(f"✅ No anomaly detected. Deviation: {deviation:.2f}%")

            except Exception as e:
                st.error(f"Anomaly check error: {e}")

# If no file uploaded show a friendly instruction
else:
    st.info("Upload your sales transactions CSV or Excel file with columns 'InvoiceDate' and 'Quantity' to begin.")

# ---------------------- END ----------------------
