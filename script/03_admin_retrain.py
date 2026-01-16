"""
03_admin_retrain.py

ADMIN-ONLY SCRIPT
-----------------
- Merge original training data with approved user-uploaded events
- Retrain logistic regression model
- Overwrite artifacts/logit_signal_model.joblib

This script is intentionally NOT a Streamlit app.
It should be run manually by the project owner.
"""

from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ======================================================
# Paths (robust, project-root based)
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_TRAIN_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed_data"
    / "event_returns_multi_company.csv"
)

USER_POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "pools"
    / "user_upload_pool.jsonl"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "logit_signal_model.joblib"


# ======================================================
# Config
# ======================================================

FEATURE_COLS = ["layoff_signal", "ai_signal"]
TARGET_COL = "up"


# ======================================================
# Loaders
# ======================================================

def load_base_training_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_TRAIN_DATA)
    df = df.dropna(subset=FEATURE_COLS + ["ret_t_plus_1"])
    df[TARGET_COL] = (df["ret_t_plus_1"] > 0).astype(int)
    return df[FEATURE_COLS + [TARGET_COL]]


def load_approved_user_events() -> pd.DataFrame:
    if not USER_POOL_PATH.exists():
        print("⚠ No user pool found. Skipping user data.")
        return pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL])

    records = []
    with open(USER_POOL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if not obj.get("approved_for_training", False):
                continue

            if obj.get("label") not in [0, 1]:
                continue

            records.append({
                "layoff_signal": obj["layoff_signal"],
                "ai_signal": obj["ai_signal"],
                "up": obj["label"]
            })

    if not records:
        print("⚠ No approved user events found.")
        return pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL])

    return pd.DataFrame(records)


# ======================================================
# Training
# ======================================================

def train_model(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000
        ))
    ])

    model.fit(X, y)
    return model


# ======================================================
# Main
# ======================================================

def main():
    print("=== ADMIN RETRAIN PIPELINE START ===")

    print("Loading base training data...")
    base_df = load_base_training_data()
    print(f"Base samples: {len(base_df)}")

    print("Loading approved user events...")
    user_df = load_approved_user_events()
    print(f"Approved user samples: {len(user_df)}")

    full_df = pd.concat([base_df, user_df], ignore_index=True)

    print(f"Total training samples: {len(full_df)}")
    print(f"Class balance (up=1 rate): {full_df[TARGET_COL].mean():.3f}")

    if len(full_df) < 20:
        raise RuntimeError("Too few samples to retrain safely.")

    print("Training model...")
    model = train_model(full_df)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model saved to: {MODEL_PATH}")
    print("=== RETRAIN COMPLETE ===")


if __name__ == "__main__":
    main()
