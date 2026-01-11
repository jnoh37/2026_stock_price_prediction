import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/processed_data/event_returns_multi_company.csv"
OUT_DIR = "artifacts"
MODEL_PATH = os.path.join(OUT_DIR, "logit_signal_model.joblib")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["layoff_signal", "ai_signal", "ret_t_plus_1"])

    df["up"] = (df["ret_t_plus_1"] > 0).astype(int)

    X = df[["layoff_signal", "ai_signal"]]
    y = df["up"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print("Training rows:", len(df))
    print("Class balance (up=1 rate):", df["up"].mean())

if __name__ == "__main__":
    main()
