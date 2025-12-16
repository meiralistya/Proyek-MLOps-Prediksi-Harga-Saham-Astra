import streamlit as st
import joblib
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime

# LOAD CONFIG
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["path"]
THRESHOLD = config["model"]["threshold"]

model = joblib.load(MODEL_PATH)

# FEATURE ENGINEERING
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_20d"] = df["Close"].pct_change(20)

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()
    df["RSI_14"] = calculate_rsi(df["Close"])

    return df

# STREAMLIT UI
st.set_page_config(page_title="Stock Prediction", layout="centered")

st.title("Prediksi Arah Harga Saham")
st.write("Model MLOps – Multi-Stock Price Direction Prediction")

ticker = st.text_input("Masukkan ticker saham (contoh: ASII.JK)", "ASII.JK")

if st.button("Prediksi"):
    with st.spinner("Mengambil data dan memprediksi..."):
        df = yf.Ticker(ticker).history(period="3mo")

        if df.empty:
            st.error("Data saham tidak ditemukan")
        else:
            df = prepare_features(df)
            df = df.dropna()

            for col in model.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0

            X = df[model.feature_names_in_].iloc[[-1]]
            prob = model.predict_proba(X)[0][1]

            prediction = "NAIK" if prob >= THRESHOLD else "TURUN"

            st.success(f"Prediksi: **{prediction}**")
            st.metric("Probabilitas Naik", f"{prob:.2%}")
            st.caption(f"⏱️ {datetime.now()}")
