import re
import numpy as np
import streamlit as st
from bertopic import BERTopic
from save_user_upload import save_user_event
from pathlib import Path

# =========================
# Path Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BERTOPIC_PATH = PROJECT_ROOT / "artifacts" / "bertopic_model"

# =========================
# Utility & Signal Extraction
# =========================
@st.cache_resource
def load_bertopic():
    return BERTopic.load(str(BERTOPIC_PATH))

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def compute_topic_signals(text: str, _model):
    # Split text into sentences for BERTopic
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
    if not sentences:
        return {}
    
    # Transform sentences to find topics
    topics, probs = _model.transform(sentences)
    
    # Calculate topic distribution (ratio of each topic in the document)
    unique_topics, counts = np.unique(topics, return_counts=True)
    total_sentences = len(sentences)
    
    # Create a dictionary of signals: { "topic_0": 0.2, "topic_5": 0.1 ... }
    signals = {f"topic_{t}": float(c / total_sentences) for t, c in zip(unique_topics, counts) if t != -1}
    return signals

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Data Intake (Topic-based)", layout="centered")
st.title("Data Intake → Topic Staging Pool")

topic_model = load_bertopic()

text = st.text_area("Paste a news / announcement excerpt:", height=260)

if st.button("Extract Topic Signals"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        txt = clean_text(text)
        signals = compute_topic_signals(txt, topic_model)

        st.subheader("Extracted Topic Composition")
        st.write(signals)

        st.session_state.ready_to_save = True
        st.session_state.txt = txt
        st.session_state.topic_signals = signals

if st.session_state.get("ready_to_save", False):
    if st.button("Save to pool"):
        # Save logic: pass the topic_signals dictionary instead of layoff/ai
        event_id = save_user_event(
            raw_text=st.session_state.txt,
            topic_signals=st.session_state.topic_signals, # Updated field
            source="streamlit_intake"
        )
        st.success(f"Saved! event_id = {event_id}")
        st.session_state.ready_to_save = False