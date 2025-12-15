import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

print("=" * 70)
print("MULTI-STOCK PRICE DIRECTION PREDICTION - TRAINING")
print("=" * 70)

# 1. CONFIGURATION
TICKERS = ["BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "ICBP.JK"]
PERIOD = "3y"
TARGET_THRESHOLD = 0.002

DATA_PATH = "data/multistock_tuning_data.csv"
MODEL_PATH = "models/final_model.pkl"
META_PATH = "models/final_model_metadata.json"
TUNED_META_PATH = "models/model_metadata.json"

os.makedirs("models", exist_ok=True)

# 2. FEATURE ENGINEERING FUNCTIONS
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

# 3. LOAD DATA (REPRODUCIBLE)
print("\nLoading training data...")
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

base_features = [
    "Return_1d", "Return_5d", "Return_20d",
    "MA_5", "MA_20", "MA_50",
    "Volume_Ratio",
    "High_Low_Range",
    "Volatility_20",
    "RSI_14"
]

df = pd.get_dummies(df, columns=["Ticker"], drop_first=True)

X = df[[c for c in df.columns if c in base_features or c.startswith("Ticker_")]]
y = df["Target"]

print(f"Samples: {len(X)} | Features: {X.shape[1]}")

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# 4. LOAD BEST PARAMETERS FROM TUNING
print("\nLoading best hyperparameters...")
with open(TUNED_META_PATH, "r") as f:
    tuned_meta = json.load(f)

best_params = tuned_meta["best_parameters"]
print("Best parameters loaded")

# 5. TRAIN FINAL MODEL
split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

print("\nTraining final model...")
model.fit(X_train, y_train)

# 6. EVALUATION
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba)
}

print("\nFINAL MODEL PERFORMANCE")
for k, v in metrics.items():
    print(f"{k.upper():10s}: {v:.4f}")

# 7. SAVE FINAL MODEL & METADATA
joblib.dump(model, MODEL_PATH)

final_metadata = {
    "model": "XGBoost",
    "stage": "final_training",
    "tickers": TICKERS,
    "metrics": metrics,
    "best_parameters": best_params,
    "features": list(X.columns),
    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_period": f"{df.index.min().date()} to {df.index.max().date()}"
}

with open(META_PATH, "w") as f:
    json.dump(final_metadata, f, indent=2)

print("\nFINAL MODEL SAVED")
print("Artifacts:")
print("- models/final_model.pkl")
print("- models/final_model_metadata.json")
print("=" * 70)