

import pandas as pd
import pickle

ACCU_CSV = "Mock_db\mock_train_accu.csv"
REDEEM_CSV = "Mock_db\mock_train_redeem.csv"

EARN_DB_PATH = "behavioral_model/earn_user_db.pkl"
REDEEM_DB_PATH = "behavioral_model/redeem_user_db.pkl"


def load_and_prepare(csv_path):
    
    df = pd.read_csv(csv_path)
    
    # Convert transaction_date
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    
    # Sort properly
    df = df.sort_values(
        by=["customer_unique_id", "transaction_date"]
    )
    
    return df


def build_user_db(df):
    
    user_db = {}
    
    for _, row in df.iterrows():
        
        customer_id = row["customer_unique_id"]
        txn_record = row.to_dict()
        
        if customer_id not in user_db:
            user_db[customer_id] = []
        
        user_db[customer_id].append(txn_record)
    
    return user_db


def main():
    
    print("Loading accrual training data...")
    accu_df = load_and_prepare(ACCU_CSV)
    
    print("Building earn user DB...")
    earn_user_db = build_user_db(accu_df)
    
    print(f"Earn users loaded: {len(earn_user_db)}")
    
    with open(EARN_DB_PATH, "wb") as f:
        pickle.dump(earn_user_db, f)
    
    print(f"Earn DB saved to {EARN_DB_PATH}")
    
    
    print("\nLoading redeem training data...")
    redeem_df = load_and_prepare(REDEEM_CSV)
    
    print("Building redeem user DB...")
    redeem_user_db = build_user_db(redeem_df)
    
    print(f"Redeem users loaded: {len(redeem_user_db)}")
    
    with open(REDEEM_DB_PATH, "wb") as f:
        pickle.dump(redeem_user_db, f)
    
    print(f"Redeem DB saved to {REDEEM_DB_PATH}")


if __name__ == "__main__":
    main()