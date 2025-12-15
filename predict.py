import yfinance as yf
import pandas as pd
import joblib
import json
import warnings
import os
warnings.filterwarnings("ignore")

print("=" * 70)
print("MULTI-STOCK PRICE DIRECTION PREDICTION - INFERENCE")
print("=" * 70)

# CONFIG
TICKERS = ["BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "ICBP.JK"]
MODEL_PATH = "models/tuned_best_model.pkl"
META_PATH = "models/model_metadata.json"
LOOKBACK_DAYS = 60
THRESHOLD = 0.5

# LOAD MODEL
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
print("Model loaded successfully")

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


# RUN PREDICTION
results = []

for ticker in TICKERS:
    print(f"\nProcessing {ticker}...")

    df = yf.Ticker(ticker).history(period=f"{LOOKBACK_DAYS}d")

    if df.empty or len(df) < LOOKBACK_DAYS:
        print(f"Not enough data for {ticker}")
        continue

    df = prepare_features(df)
    df = df.dropna()

    # === FIX FEATURE MISMATCH ===
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURES].iloc[[-1]]
    prob = model.predict_proba(X)[0][1]

    results.append({
        "ticker": ticker,
        "prediction": "UP" if prob >= THRESHOLD else "DOWN",
        "probability_up": round(prob, 4),
        "confidence": "HIGH" if abs(prob - 0.5) > 0.15 else "LOW"
    })

results_df = pd.DataFrame(results)
print("\nPREDICTION RESULTS")
print(results_df)

os.makedirs("predictions", exist_ok=True)
results_df.to_csv("predictions/latest_predictions.csv", index=False)

print("\nSaved to predictions/latest_predictions.csv")
print("=" * 70)