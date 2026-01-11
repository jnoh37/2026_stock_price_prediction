import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# =========================
# 1. Load data
# =========================

DATA_PATH = "data/processed_data/event_returns_multi_company.csv"
df = pd.read_csv(DATA_PATH)

# =========================
# 2. Construct label
# =========================

# Binary label: positive abnormal return at t+1
df["up"] = (df["ret_t_plus_1"] > 0).astype(int)

# Drop rows with missing values
df = df.dropna(subset=["layoff_signal", "ai_signal", "ret_t_plus_1"])

print("Number of observations:", len(df))
print(df[["layoff_signal", "ai_signal", "ret_t_plus_1", "up"]])


# =========================
# 3. Define X and y
# =========================

X = df[["layoff_signal", "ai_signal"]]
y = df["up"]


# =========================
# 4. Logistic regression pipeline
# =========================
# Scaling + Logistic Regression
# (scaling helps coefficient comparability)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(
        penalty="l2",
        solver="lbfgs"
    ))
])

model.fit(X, y)


# =========================
# 5. Inspect coefficients
# =========================

coef = model.named_steps["logit"].coef_[0]
intercept = model.named_steps["logit"].intercept_[0]
feature_names = X.columns

print("\nLogistic Regression Coefficients:")
for name, c in zip(feature_names, coef):
    print(f"{name:15s}: {c:.3f}")

print(f"Intercept        : {intercept:.3f}")


# =========================
# 6. In-sample evaluation
# =========================

y_pred = model.predict(X)
print("\nClassification Report:")
print(classification_report(y, y_pred, digits=3))


# =========================
# 7. Predicted probabilities (optional)
# =========================

df["prob_up"] = model.predict_proba(X)[:, 1]
print("\nPredicted probabilities:")
print(df[["company", "layoff_signal", "ai_signal", "prob_up"]])
