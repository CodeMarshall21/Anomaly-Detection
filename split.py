import pandas as pd
from sklearn.model_selection import train_test_split

accu_df = pd.read_csv("Datasets/synthetic_accural_data.csv")
redm_df = pd.read_csv("Datasets/synthetic_redeem_data.csv")


accu_train, accu_test = train_test_split(
    accu_df,
    test_size=0.2,
    stratify=accu_df["is_anomaly"],
    random_state=42
)

# accu_test_actual = accu_test["is_anomaly"]
# accu_test = accu_test.drop(columns=["is_anomaly"])

accu_train_actual = accu_train["is_anomaly"]
accu_train = accu_train.drop(columns=["is_anomaly"])

# accu_test.to_csv("payload/accu_test_payload.csv", index=False)
accu_train.to_csv("payload/accu_train_payload.csv", index=False)

# print("Accrual test CSV created.")
print("Accrual train CSV created.")


redm_train, redm_test = train_test_split(
    redm_df,
    test_size=0.2,
    stratify=redm_df["is_anomaly"],
    random_state=42
)

# redm_test_actual = redm_test["is_anomaly"]
# redm_test = redm_test.drop(columns=["is_anomaly"])

redm_train_actual = redm_train["is_anomaly"]
redm_train = redm_train.drop(columns=["is_anomaly"])

# redm_test.to_csv("payload/redm_test_payload.csv", index=False)
redm_train.to_csv("payload/redm_train_payload.csv", index=False)

# print("Redemption test CSV created.")
print("Redemption train CSV created.")
