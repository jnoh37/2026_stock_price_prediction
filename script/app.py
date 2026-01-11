import re
import numpy as np
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = "artifacts/logit_signal_model.joblib"

# --- same keyword sets as your pipeline ---
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

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def compute_signals(text: str):
    """
    Compute layoff_signal and ai_signal using restricted TF-IDF over a single document.
    For a single doc, TF-IDF degenerates to TF with normalization, but it's fine for scoring.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        vocabulary=VOCAB,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )
    X = vectorizer.fit_transform([text])  # shape (1, |VOCAB|)
    feature_names = vectorizer.get_feature_names_out()

    layoff_idx = [i for i, f in enumerate(feature_names) if f in LAYOFF_KEYWORDS]
    ai_idx = [i for i, f in enumerate(feature_names) if f in AI_KEYWORDS]

    layoff_signal = float(np.asarray(X[:, layoff_idx].sum(axis=1)).ravel()[0]) if layoff_idx else 0.0
    ai_signal = float(np.asarray(X[:, ai_idx].sum(axis=1)).ravel()[0]) if ai_idx else 0.0

    return layoff_signal, ai_signal

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Event Signal → Stock Move Probability", layout="centered")
st.title("News/Announcement → Up/Down Probability")
st.caption("Uses layoff & AI text signals + Logistic Regression (trained on your aligned event dataset).")

model = load_model()

text = st.text_area(
    "Paste a news/announcement text here:",
    height=220,
    placeholder="Example: The bank announced a restructuring plan and will cut 2,000 jobs..."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            txt = clean_text(text)
            layoff_signal, ai_signal = compute_signals(txt)

            X_in = np.array([[layoff_signal, ai_signal]])
            prob_up = float(model.predict_proba(X_in)[0, 1])
            prob_down = 1.0 - prob_up

            st.subheader("Result")
            st.metric("Up probability", f"{prob_up:.2%}")
            st.metric("Down probability", f"{prob_down:.2%}")

            st.subheader("Signals")
            st.write({"layoff_signal": layoff_signal, "ai_signal": ai_signal})

with col2:
    st.markdown("### Notes")
    st.write(
        "- This is an interpretable, hypothesis-driven NLP model.\n"
        "- Probability reflects patterns in your training dataset (small sample).\n"
        "- For better quality, expand event samples and add market-adjusted labels."
    )
