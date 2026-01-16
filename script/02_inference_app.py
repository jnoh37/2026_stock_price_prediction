import re
import numpy as np
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# =====================================================
# Resolve project root & model path (ROBUST)
# =====================================================

# 当前文件：.../script/02_inference_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "logit_signal_model.joblib"


# =====================================================
# Keyword vocab (must match training pipeline)
# =====================================================

LAYOFF_KEYWORDS = [
    "layoff", "layoffs", "job", "jobs", "cut", "cuts",
    "headcount", "workforce", "redundancy",
    "restructuring", "downsizing",
    "cost", "costs", "expense", "expenses",
    "efficiency", "efficiencies"
]

AI_KEYWORDS = [
    "ai", "artificial", "intelligence",
    "automation", "automated",
    "machine", "learning",
    "generative",
    "productivity",
    "digital", "transformation"
]

VOCAB = sorted(set(LAYOFF_KEYWORDS + AI_KEYWORDS))


# =====================================================
# Utilities
# =====================================================

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_signals(text: str):
    """
    Compute layoff_signal and ai_signal using restricted TF-IDF over a single document.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        vocabulary=VOCAB,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )

    X = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    layoff_idx = [i for i, f in enumerate(feature_names) if f in LAYOFF_KEYWORDS]
    ai_idx = [i for i, f in enumerate(feature_names) if f in AI_KEYWORDS]

    layoff_signal = float(X[:, layoff_idx].sum()) if layoff_idx else 0.0
    ai_signal = float(X[:, ai_idx].sum()) if ai_idx else 0.0

    return layoff_signal, ai_signal


# =====================================================
# Model loader
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at:\n{MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(
    page_title="Inference → Up/Down Probability",
    layout="centered"
)

st.title("Inference → Up/Down Probability")
st.caption(
    "This app ONLY runs predictions using a pre-trained model. "
    "It does NOT save user inputs to the training pool."
)

model = load_model()

text = st.text_area(
    "Paste a news / announcement / conference-call excerpt:",
    height=260,
    placeholder="Example: The bank announced a restructuring plan and will cut 2,000 jobs..."
)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        txt = clean_text(text)
        layoff_signal, ai_signal = compute_signals(txt)

        X_in = np.array([[layoff_signal, ai_signal]])
        prob_up = float(model.predict_proba(X_in)[0, 1])
        prob_down = 1.0 - prob_up

        st.subheader("Prediction")
        st.metric("Up probability", f"{prob_up:.2%}")
        st.metric("Down probability", f"{prob_down:.2%}")

        st.subheader("Signals")
        st.write({
            "layoff_signal": layoff_signal,
            "ai_signal": ai_signal
        })

st.markdown("### Notes")
st.write(
    "- Model loaded from project-level `artifacts/` directory.\n"
    "- Training and inference environments must match.\n"
    "- Accuracy improves mainly through better labels and more curated events."
)