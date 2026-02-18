from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import pickle
import io
import os

from catboost import CatBoostClassifier, Pool

app = FastAPI()

MAX_ROWS = 2000
MODEL_DIR = "model"

# -----------------------------
# Load Model Artifacts
# -----------------------------
def load_artifacts(prefix):

    model = CatBoostClassifier()
    model.load_model(f"{MODEL_DIR}/{prefix}_model.cbm")

    with open(f"{MODEL_DIR}/{prefix}_cat_cols.pkl", "rb") as f:
        cat_cols = pickle.load(f)

    with open(f"{MODEL_DIR}/{prefix}_feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    with open(f"{MODEL_DIR}/{prefix}_decile_bins.pkl", "rb") as f:
        decile_bins = pickle.load(f)

    with open(f"{MODEL_DIR}/{prefix}_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    threshold = metrics["threshold"]

    return model, cat_cols, feature_cols, decile_bins, threshold


accu_model, accu_cat_cols, accu_feature_cols, accu_bins, accu_threshold = load_artifacts("accu")
redm_model, redm_cat_cols, redm_feature_cols, redm_bins, redm_threshold = load_artifacts("redm")


# -----------------------------
# Feature Engineering
# -----------------------------
columns_to_drop = [
    'transaction_id',
    'customer_unique_id',
    'anomaly_types',
    'created_at',
    'updated_at',
    'bill_date',
    'business_date',
    'rule_id',
    'rule_type',
    'item_code',
    'sku_category_code',
    'category_code',
    'store_id',
    'activity_id',
    'funding_partner_id'
]

def feature_engineering(df):

    df = df.drop(columns=columns_to_drop, errors="ignore")

    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    df['hour'] = df['transaction_date'].dt.hour
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df = df.drop(columns=['transaction_date'])

    return df


# -----------------------------
# Decile Logic
# -----------------------------
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


# -----------------------------
# CSV Endpoint
# -----------------------------
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum allowed rows is {MAX_ROWS}."
        )

    if "transaction_type" not in df.columns:
        raise HTTPException(status_code=400, detail="transaction_type missing.")

    if df["transaction_type"].nunique() > 1:
        raise HTTPException(status_code=400, detail="Mixed transaction types not allowed.")

    transaction_type = df["transaction_type"].iloc[0].lower()

    if "customer_unique_id" not in df.columns:
        raise HTTPException(status_code=400, detail="customer_unique_id missing.")

    customer_ids = df["customer_unique_id"]

    df_processed = feature_engineering(df.copy())

    # -----------------------------
    # Model Routing
    # -----------------------------
    if transaction_type == "earn":
        model = accu_model
        bins = accu_bins
        cat_cols = accu_cat_cols
        feature_cols = accu_feature_cols
        threshold = accu_threshold
        inference_type = "accrual"
    elif transaction_type == "redeem":
        model = redm_model
        bins = redm_bins
        cat_cols = redm_cat_cols
        feature_cols = redm_feature_cols
        threshold = redm_threshold
        inference_type = "redemption"
    else:
        raise HTTPException(status_code=400, detail="Invalid transaction_type.")

    # -----------------------------
    # Enforce Feature Order
    # -----------------------------
    missing_cols = set(feature_cols) - set(df_processed.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_cols}"
        )

    df_processed = df_processed[feature_cols]

    # -----------------------------
    # Prediction
    # -----------------------------
    pool = Pool(
        data=df_processed,
        cat_features=cat_cols
    )

    y_prob = model.predict_proba(pool)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    results = []

    for i in range(len(df)):
        decile = assign_decile(y_prob[i], bins)

        results.append({
            "customer_id": customer_ids.iloc[i],
            "prediction": int(y_pred[i]),
            "probability": float(y_prob[i]),
            "decile": decile,
            "anomaly_severity": assign_severity(decile)
        })

    return {
        "inference_type": inference_type,
        "total_customers": len(df),
        "results": results
    }
