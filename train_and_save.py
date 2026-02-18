import pandas as pd
import numpy as np
import pickle
import os

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# -----------------------------
# Config
# -----------------------------
THRESHOLD = 0.3
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

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
# Preprocessing
# -----------------------------
def preprocess(df):
    df = df.drop(columns=columns_to_drop, errors="ignore")

    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    df['hour'] = df['transaction_date'].dt.hour
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df = df.drop(columns=['transaction_date'])

    return df


# -----------------------------
# Training Pipeline
# -----------------------------
def train_model(df, model_prefix):

    df = preprocess(df)

    y = df["is_anomaly"]
    X = df.drop(columns=["is_anomaly"])

    # Save feature order explicitly
    feature_cols = X.columns.tolist()
    with open(f"{MODEL_DIR}/{model_prefix}_feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    # Detect categorical columns
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    with open(f"{MODEL_DIR}/{model_prefix}_cat_cols.pkl", "wb") as f:
        pickle.dump(cat_cols, f)

    # -----------------------------
    # Train / Validation Split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Handle class imbalance
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = CatBoostClassifier(
        class_weights=[1, scale_weight],
        verbose=0
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols
    )

    # Save model
    model.save_model(f"{MODEL_DIR}/{model_prefix}_model.cbm")

    # -----------------------------
    # Validation Evaluation
    # -----------------------------
    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob > THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_val, val_prob)
    pr_auc = average_precision_score(y_val, val_prob)
    f1 = f1_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    cm = confusion_matrix(y_val, val_pred).tolist()

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_at_0.3": f1,
        "precision_at_0.3": precision,
        "recall_at_0.3": recall,
        "confusion_matrix_at_0.3": cm,
        "threshold": THRESHOLD
    }

    with open(f"{MODEL_DIR}/{model_prefix}_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # -----------------------------
    # Decile Bins from Validation
    # -----------------------------
    decile_bins = np.percentile(val_prob, [10,20,30,40,50,60,70,80,90])

    with open(f"{MODEL_DIR}/{model_prefix}_decile_bins.pkl", "wb") as f:
        pickle.dump(decile_bins, f)

    print(f"{model_prefix.upper()} model trained and saved.")
    print("Validation Metrics:")
    print(metrics)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":

    print("Loading datasets...")

    accu_df = pd.read_csv("Datasets/synthetic_accural_data.csv")
    redm_df = pd.read_csv("Datasets/synthetic_redeem_data.csv")

    print("Training Accrual Model...")
    train_model(accu_df, "accu")

    print("\nTraining Redemption Model...")
    train_model(redm_df, "redm")

    print("\nAll models trained successfully.")
