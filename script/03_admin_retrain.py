import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = PROJECT_ROOT / "data" / "processed_data" / "topic_returns_combined.csv"
USER_POOL = PROJECT_ROOT / "data" / "pools" / "user_upload_pool.jsonl" # Assuming this format
MODEL_PATH = PROJECT_ROOT / "artifacts" / "topic_logit_model.joblib"
FEATURE_COLS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.joblib"

def main():
    # 1. Load Base Data
    df_base = pd.read_csv(BASE_DATA)
    topic_cols = [c for c in df_base.columns if c.startswith("topic_")]
    
    # 2. Load and merge User Data (Logic similar to your original script)
    # Note: user_df must have columns matching 'topic_0', 'topic_1' etc.
    # ... (Load user_df logic) ...
    
    # full_df = pd.concat([df_base, user_df])
    full_df = df_base.dropna(subset=['ret_t_plus_1'])
    full_df['up'] = (full_df['ret_t_plus_1'] > 0).astype(int)

    # 3. Train
    X = full_df[topic_cols]
    y = full_df['up']
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression())
    ])
    model.fit(X, y)
    
    # 4. Save both Model and Column Order
    joblib.dump(model, MODEL_PATH)
    joblib.dump(topic_cols, FEATURE_COLS_PATH) # CRITICAL: save feature order
    print(f"Retrained model saved with {len(topic_cols)} features.")

if __name__ == "__main__":
    main()