import pandas as pd
import numpy as np
import pickle
import os

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

MODEL_DIR = "model"
OUTPUT_DIR = "batch_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# Columns to Drop
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


# -----------------------------
# Feature Engineering
# -----------------------------
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
# Load Artifacts
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


# -----------------------------
# Test Evaluation
# -----------------------------
def evaluate_on_test(input_path):

    df = pd.read_csv(input_path)

    if df["transaction_type"].nunique() > 1:
        raise ValueError("Mixed transaction types not allowed.")

    transaction_type = df["transaction_type"].iloc[0].lower()

    if transaction_type == "earn":
        prefix = "accu"
        inference_type = "accrual"
    elif transaction_type == "redeem":
        prefix = "redm"
        inference_type = "redemption"
    else:
        raise ValueError("Invalid transaction_type.")

    # Split dataset
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["is_anomaly"],
        random_state=RANDOM_STATE
    )

    print(f"Total rows: {len(df)}")
    print(f"Test rows: {len(test_df)}")

    actual_values = test_df["is_anomaly"]

    model, cat_cols, feature_cols, bins, threshold = load_artifacts(prefix)

    test_processed = feature_engineering(test_df.copy())
    test_processed = test_processed.drop(columns=["is_anomaly"])

    # Enforce feature order
    missing_cols = set(feature_cols) - set(test_processed.columns)
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    test_processed = test_processed[feature_cols]

    pool = Pool(
        data=test_processed,
        cat_features=cat_cols
    )

    probabilities = model.predict_proba(pool)[:, 1]
    predictions = (probabilities > threshold).astype(int)

    deciles = [assign_decile(p, bins) for p in probabilities]
    severities = [assign_severity(d) for d in deciles]

    # Create final output
    output_df = test_df.copy()
    output_df["actual_is_anomaly"] = actual_values
    output_df["predicted_is_anomaly"] = predictions
    output_df["probability"] = probabilities
    output_df["decile"] = deciles
    output_df["anomaly_severity"] = severities
    output_df["inference_type"] = inference_type

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_test_scored.csv"
    )

    output_df.to_csv(output_file, index=False)

    print(f"Saved test evaluation to: {output_file}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    # evaluate_on_test("Datasets/synthetic_accural_data.csv")
    # or
    evaluate_on_test("Datasets/synthetic_redeem_data.csv")
