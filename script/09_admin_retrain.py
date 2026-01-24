import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[0] # Updated for flat directory
BASE_DATA = PROJECT_ROOT / "data" / "processed_data" / "topic_returns_combined.csv"
USER_POOL = PROJECT_ROOT / "data" / "pools" / "user_upload_pool.jsonl"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "topic_logit_model.joblib"
FEATURE_COLS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.joblib"

def load_user_pool():
    """Load and flatten signals from the user pool JSONL."""
    if not USER_POOL.exists():
        return pd.DataFrame()
    
    records = []
    with open(USER_POOL, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Only use data that has been manually approved
            if data.get("approved_for_training") is True:
                # Flatten the 'signals' dictionary into the main record
                flat_record = {**data["signals"]}
                records.append(flat_record)
    return pd.DataFrame(records)

def main():
    # 1. Load Base Data
    df_base = pd.read_csv(BASE_DATA)
    
    # 2. Load and merge User Data
    user_df = load_user_pool()
    
    if not user_df.empty:
        full_df = pd.concat([df_base, user_df], ignore_index=True)
        print(f"Merged {len(user_df)} new approved records.")
    else:
        full_df = df_base
        print("No new approved user data found. Using base data only.")

    full_df = full_df.dropna(subset=['ret_t_plus_1'])
    full_df['up'] = (full_df['ret_t_plus_1'] > 0).astype(int)

    # 3. Define Hybrid Features (Topics + Signals)
    topic_cols = [c for c in full_df.columns if c.startswith("topic_")]
    X_cols = topic_cols + ["layoff_signal", "ai_signal"]
    
    X = full_df[X_cols]
    y = full_df['up']
    
    # 4. Train with Scaling
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000))
    ])
    model.fit(X, y)
    
    # 5. Save both Model and Column Order
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X_cols, FEATURE_COLS_PATH) 
    print(f"Retrained hybrid model saved with {len(X_cols)} features.")

if __name__ == "__main__":
    main()