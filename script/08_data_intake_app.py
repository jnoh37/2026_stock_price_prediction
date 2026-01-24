import streamlit as st
import numpy as np
import re
from bertopic import BERTopic
from pathlib import Path
from save_user_upload import save_user_event

# =========================
# 1. Config & Signals
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[0]
BERTOPIC_PATH = PROJECT_ROOT / "artifacts" / "bertopic_model"

LAYOFF_KEYWORDS = ["layoff", "layoffs", "job", "jobs", "cut", "cuts", "headcount", "workforce", "redundancy", "restructuring", "downsizing", "cost", "costs", "expense", "expenses", "efficiency", "efficiencies"]
AI_KEYWORDS = ["ai", "artificial", "intelligence", "automation", "automated", "machine", "learning", "generative", "productivity", "digital", "transformation"]

# =========================
# 2. Logic & Extraction
# =========================
@st.cache_resource
def load_bertopic():
    return BERTopic.load(str(BERTOPIC_PATH)) if BERTOPIC_PATH.exists() else None

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def compute_hybrid_signals(text: str, _model):
    words = text.lower().split()
    l_score = sum(1 for w in words if w in LAYOFF_KEYWORDS)
    a_score = sum(1 for w in words if w in AI_KEYWORDS)
    keyword_signals = {"layoff_signal": float(l_score), "ai_signal": float(a_score)}

    # Sentence-level topic inference
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
    if not sentences or _model is None:
        return keyword_signals, {}
    
    topics, _ = _model.transform(sentences)
    unique_topics, counts = np.unique(topics, return_counts=True)
    total = len(sentences)
    
    # Exclude outlier topic (-1)
    topic_signals = {f"topic_{t}": float(c / total) for t, c in zip(unique_topics, counts) if t != -1}
    
    return keyword_signals, topic_signals

# =========================
# 3. Streamlit UI
# =========================
st.set_page_config(page_title="Data Intake (Hybrid)", layout="centered")
st.title("📥 Data Intake → Hybrid Staging Pool")

topic_model = load_bertopic()

if "ready_to_save" not in st.session_state:
    st.session_state.ready_to_save = False

text_input = st.text_area("Paste a news / announcement excerpt:", height=260)

if st.button("Extract Hybrid Signals"):
    if not text_input.strip():
        st.warning("Please paste some text.")
    else:
        cleaned_txt = clean_text(text_input)
        kw_sig, top_sig = compute_hybrid_signals(cleaned_txt, topic_model)

        col1, col2 = st.columns(2)
        col1.json(kw_sig)
        col2.json(top_sig)

        st.session_state.ready_to_save = True
        st.session_state.txt = cleaned_txt
        st.session_state.all_signals = {**kw_sig, **top_sig}

if st.session_state.ready_to_save:
    st.divider()
    if st.button("Confirm & Save to Pool"):
        try:
            save_user_event(st.session_state.txt, st.session_state.all_signals)
            st.success("Saved successfully.")
            st.session_state.ready_to_save = False
        except Exception as e:
            st.error(f"Save failed: {e}")