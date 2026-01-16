import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from save_user_upload import save_user_event

# =========================
# Keyword vocab (must match pipeline)
# =========================

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

# =========================
# Utility
# =========================

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_signals(text: str):
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


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Data Intake (Staging Pool)", layout="centered")
st.title("Data Intake → Staging Pool")
st.caption(
    "This app ONLY collects user-submitted events into a staging pool. "
    "It does NOT retrain models and does NOT run predictions."
)

source = st.selectbox(
    "What is the source of this text?",
    options=["news", "announcement", "conference_call", "other"],
    index=0
)

text = st.text_area(
    "Paste a news / announcement / conference-call excerpt:",
    height=260,
    placeholder="Paste the raw text here..."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Extract signals"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            txt = clean_text(text)
            layoff_signal, ai_signal = compute_signals(txt)

            st.subheader("Extracted signals")
            st.write({"layoff_signal": layoff_signal, "ai_signal": ai_signal})

            # store in session_state for saving step
            st.session_state.ready_to_save = True
            st.session_state.txt = txt
            st.session_state.layoff_signal = layoff_signal
            st.session_state.ai_signal = ai_signal

with col2:
    st.markdown("### What happens here?")
    st.write(
        "- We compute two interpretable signals: **layoff** and **AI**.\n"
        "- If you save, the record is appended to a **staging pool** (JSONL).\n"
        "- Retraining is **manual/offline** and controlled by the maintainer."
    )

st.markdown("---")

# Save section (only after extraction)
if st.session_state.get("ready_to_save", False):
    if st.button("Save to pool"):
        event_id = save_user_event(
            raw_text=st.session_state.txt,
            layoff_signal=st.session_state.layoff_signal,
            ai_signal=st.session_state.ai_signal,
            source=f"streamlit_intake:{source}"
        )
        st.success(f"Saved to staging pool. event_id = {event_id}")
        st.session_state.ready_to_save = False
else:
    st.info("Step 1: click **Extract signals**. Then you can save the record to the pool.")
