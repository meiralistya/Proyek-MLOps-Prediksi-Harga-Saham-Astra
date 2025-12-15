import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

print("=" * 70)
print("MULTI-STOCK PRICE DIRECTION PREDICTION - TUNING")
print("=" * 70)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
TICKERS = ["BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "ICBP.JK"]
PERIOD = "3y"
TARGET_THRESHOLD = 0.002  # 0.2%

DATA_PATH = "data/multistock_tuning_data.csv"
MODEL_PATH = "models/tuned_best_model.pkl"
META_PATH = "models/model_metadata.json"

# --------------------------------------------------
# 2. FEATURE ENGINEERING FUNCTIONS
# --------------------------------------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def load_data():
    try:
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        print(f"Loaded cached data: {df.shape}")
        return df
    except:
        print("Downloading multi-stock data...")

    dfs = []

    for ticker in TICKERS:
        stock = yf.Ticker(ticker)
        df = stock.history(period=PERIOD)

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

        df["Future_Return"] = df["Close"].pct_change().shift(-1)
        df["Target"] = (df["Future_Return"] > TARGET_THRESHOLD).astype(int)

        df["Ticker"] = ticker
        df = df.dropna()

        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all.to_csv(DATA_PATH)

    print(f"Created dataset: {df_all.shape}")
    return df_all


df = load_data()

# --------------------------------------------------
# 3. PREPARE FEATURES
# --------------------------------------------------
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

print("\nDATA SUMMARY")
print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"UP: {y.mean():.2%} | DOWN: {(1 - y.mean()):.2%}")

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# 4. TIME SERIES SPLIT
tscv = TimeSeriesSplit(n_splits=5)

# 5. XGBOOST TUNING
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "min_child_weight": [3, 5],
    "gamma": [0, 0.1],
    "reg_alpha": [0.1, 0.5],
    "reg_lambda": [1, 2]
}

print("\nStarting hyperparameter tuning...")
start_time = time.time()

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=40,
    cv=tscv,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X, y)

print(f"Tuning finished in {(time.time() - start_time) / 60:.1f} minutes")

best_params = search.best_params_

# 6. FINAL TRAIN & EVALUATION
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

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nFINAL MODEL PERFORMANCE")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# 7. SAVE MODEL & METADATA
joblib.dump(model, MODEL_PATH)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

feature_importance.to_csv("models/feature_importance.csv", index=False)

metadata = {
    "model": "XGBoost",
    "type": "Multi-Stock Direction Prediction",
    "tickers": TICKERS,
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    },
    "best_parameters": best_params,
    "features": list(X.columns),
    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_period": f"{df.index.min().date()} to {df.index.max().date()}"
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nMODEL & METADATA SAVED")
print("Artifacts:")
print("- models/tuned_best_model.pkl")
print("- models/feature_importance.csv")
print("- models/model_metadata.json")
print("=" * 70)
