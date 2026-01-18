import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# =========================
# 1. Load Data & Dictionary
# =========================
DATA_PATH = "data/processed_data/topic_returns_combined.csv"
DICT_PATH = "data/processed_data/topic_dictionary.json"

df = pd.read_csv(DATA_PATH)

with open(DICT_PATH, "r", encoding="utf-8") as f:
    topic_dict = json.load(f)

# =========================
# 2. Define Target & Features
# =========================
df["up"] = (df["ret_t_plus_1"] > 0).astype(int)
df = df.dropna(subset=["ret_t_plus_1"])

X_cols = [c for c in df.columns if c.startswith("topic_")]
X = df[X_cols]
y = df["up"]

# =========================
# 3. Model Pipeline
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(penalty="l2", solver="lbfgs", random_state=42))
])
model.fit(X, y)

# Save artifacts for Streamlit
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/topic_logit_model.joblib")
joblib.dump(X_cols, "artifacts/feature_columns.joblib")

# =========================
# 4. Inspect Coefficients with Keywords
# =========================
coef = model.named_steps["logit"].coef_[0]

impact_list = []
for name, c in zip(X_cols, coef):
    # Extract ID from 'topic_23' -> '23'
    topic_id = name.replace("topic_", "")
    # Look up keywords in our dictionary
    keywords = topic_dict.get(topic_id, "Unknown_Topic")
    
    impact_list.append({
        "Topic_ID": name,
        "Keywords": keywords,
        "Coefficient": c
    })

# Convert to DataFrame for easier sorting
impact_df = pd.DataFrame(impact_list).sort_values(by="Coefficient", ascending=False)

print("\n--- Top Topics Driving Stock Price UP ---")
print(impact_df.head(5).to_string(index=False))

print("\n--- Top Topics Driving Stock Price DOWN ---")
print(impact_df.tail(5).to_string(index=False))

# Evaluation
y_pred = model.predict(X)
print("\nClassification Report:")
print(classification_report(y, y_pred, digits=3))