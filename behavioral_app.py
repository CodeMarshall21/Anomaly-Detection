# ============================================
# behavioral_app.py
# ============================================

from fastapi import FastAPI, HTTPException
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

app = FastAPI()

MODEL_DIR = "behavioral_model"

# =====================================================
# LOAD ARTIFACTS
# =====================================================

def load_artifacts(prefix):

    model = CatBoostClassifier()
    model.load_model(f"{MODEL_DIR}/{prefix}_model.cbm")

    with open(f"{MODEL_DIR}/{prefix}_feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    with open(f"{MODEL_DIR}/{prefix}_cat_cols.pkl", "rb") as f:
        cat_cols = list(pickle.load(f))

    with open(f"{MODEL_DIR}/{prefix}_decile_bins.pkl", "rb") as f:
        decile_bins = pickle.load(f)

    with open(f"{MODEL_DIR}/{prefix}_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    threshold = metrics["threshold"]

    return model, feature_cols, cat_cols, decile_bins, threshold


accu_model, accu_feature_cols, accu_cat_cols, accu_bins, accu_threshold = load_artifacts("accu_behavioral")
redeem_model, redeem_feature_cols, redeem_cat_cols, redeem_bins, redeem_threshold = load_artifacts("redeem_behavioral")


# =====================================================
# LOAD HISTORY DATABASES
# =====================================================

with open(f"{MODEL_DIR}/earn_user_db.pkl", "rb") as f:
    earn_history_store = pickle.load(f)

with open(f"{MODEL_DIR}/redeem_user_db.pkl", "rb") as f:
    redeem_history_store = pickle.load(f)


# =====================================================
# DECILE + SEVERITY
# =====================================================

def assign_decile(prob, bins):
    for i, cutoff in enumerate(bins):
        if prob <= cutoff:
            return f"D{i+1}"
    return "D10"


def assign_severity(decile):
    if decile == "D10":
        return "Very High Risk"
    elif decile in ["D8", "D9"]:
        return "High Risk"
    elif decile in ["D5", "D6", "D7"]:
        return "Medium Risk"
    else:
        return "Low Risk"


# =====================================================
# BASIC PREPROCESS (RUNTIME VERSION)
# =====================================================

def runtime_basic_preprocess(txn):

    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

    txn["hour"] = txn["transaction_date"].hour
    txn["day_of_week"] = txn["transaction_date"].dayofweek
    txn["is_weekend"] = int(txn["day_of_week"] in [5, 6])

    return txn


# =====================================================
# BEHAVIORAL FEATURE ENGINEERING (RUNTIME)
# =====================================================

def compute_behavioral_features(history, txn):

    now = txn["transaction_date"]

    last_1h = [
        h for h in history
        if now - h["transaction_date"] <= timedelta(hours=1)
    ]

    last_24h = [
        h for h in history
        if now - h["transaction_date"] <= timedelta(hours=24)
    ]

    txn_last_1h = len(last_1h)
    txn_last_24h = len(last_24h)

    if history:
        last_txn_time = history[-1]["transaction_date"]
        time_since_last = (now - last_txn_time).total_seconds()

        amounts = [h["amount"] for h in history]
        avg_amount = np.mean(amounts)
        std_amount = np.std(amounts)
    else:
        time_since_last = -1
        avg_amount = 0
        std_amount = 0

    amount_ratio = txn["amount"] / avg_amount if avg_amount > 0 else 0

    if std_amount > 0:
        z_score = (txn["amount"] - avg_amount) / std_amount
    else:
        z_score = 0

    balance_ratio = txn["points"] / (txn["opening_balance"] + 1e-6)
    balance_ratio = np.clip(balance_ratio, 0, 10)

    return {
        "time_since_last_transaction": time_since_last,
        "transactions_last_1h": txn_last_1h,
        "transactions_last_24h": txn_last_24h,
        "user_avg_amount": avg_amount,
        "amount_to_user_avg_ratio": amount_ratio,
        "z_score_amount": z_score,
        "balance_depletion_ratio": balance_ratio
    }


# =====================================================
# FINALIZE FEATURES (MATCH TRAINING)
# =====================================================

def finalize_for_inference(df, feature_cols):

    # Drop columns that were removed during training
    df = df.drop(columns=["transaction_date", "customer_unique_id"], errors="ignore")

    # Add missing columns
    missing = set(feature_cols) - set(df.columns)
    if missing:
        for col in missing:
            df[col] = 0

    # Strict column order
    df = df[feature_cols]

    return df


# =====================================================
# MAIN ENDPOINT
# =====================================================

@app.post("/predict-single")
def predict_single(payload: dict):

    if "transaction_type" not in payload:
        raise HTTPException(status_code=400, detail="transaction_type missing")

    if "customer_unique_id" not in payload:
        raise HTTPException(status_code=400, detail="customer_unique_id missing")

    txn_type = payload["transaction_type"].lower()
    customer_id = payload["customer_unique_id"]

    if "transaction_date" not in payload:
        raise HTTPException(status_code=400, detail="transaction_date missing")

    # 1️⃣ Basic preprocessing
    txn = runtime_basic_preprocess(payload.copy())

    # 2️⃣ Route to correct model
    if txn_type == "earn":
        model = accu_model
        feature_cols = accu_feature_cols
        cat_cols = accu_cat_cols
        bins = accu_bins
        threshold = accu_threshold
        history_store = earn_history_store

    elif txn_type == "redeem":
        model = redeem_model
        feature_cols = redeem_feature_cols
        cat_cols = redeem_cat_cols
        bins = redeem_bins
        threshold = redeem_threshold
        history_store = redeem_history_store

    else:
        raise HTTPException(status_code=400, detail="Invalid transaction_type")

    history = history_store.get(customer_id, [])

    # 3️⃣ Behavioral features
    behavioral_features = compute_behavioral_features(history, txn)

    enriched_txn = {**txn, **behavioral_features}

    df = pd.DataFrame([enriched_txn])

    # 4️⃣ Finalize features (match training)
    df = finalize_for_inference(df, feature_cols)

    pool = Pool(df, cat_features=cat_cols)

    prob = model.predict_proba(pool)[0][1]
    pred = int(prob > threshold)

    decile = assign_decile(prob, bins)
    severity = assign_severity(decile)

    # 5️⃣ Update history AFTER prediction
    history_store.setdefault(customer_id, []).append(txn)

    return {
        "customer_id": customer_id,
        "transaction_type": txn_type,
        "prediction": pred,
        "probability": float(prob),
        "decile": decile,
        "severity": severity
    }