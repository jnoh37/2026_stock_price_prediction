import streamlit as st
import joblib
import numpy as np
import pandas as pd
from bertopic import BERTopic
from pathlib import Path
import json

# Define Signal Vocabularies for Real-time scoring
LAYOFF_KEYWORDS = ["layoff", "layoffs", "job", "jobs", "cut", "cuts", "headcount", "workforce", "redundancy", "restructuring", "downsizing", "cost", "costs", "expense", "expenses", "efficiency", "efficiencies"]
AI_KEYWORDS = ["ai", "artificial", "intelligence", "automation", "automated", "machine", "learning", "generative", "productivity", "digital", "transformation"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BERTOPIC_PATH = PROJECT_ROOT / "artifacts" / "bertopic_model"
DICT_PATH = PROJECT_ROOT / "data" / "processed_data" / "topic_dictionary.json"
LOGIT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "topic_logit_model.joblib"
FEATURE_COLS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.joblib"

@st.cache_resource
def load_artifacts():
    topic_model = BERTopic.load(str(BERTOPIC_PATH))
    logit_model = joblib.load(LOGIT_MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH) 
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        topic_dict = json.load(f)
    return topic_model, logit_model, feature_cols, topic_dict

st.title("Stock Price Prediction")
topic_model, logit_model, feature_cols, topic_dict = load_artifacts()

text = st.text_area("Enter news text for prediction:", height=260)

if st.button("Predict!"):
    if text.strip():
        # 1. Topic Distribution Calculation
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
        topics, _ = topic_model.transform(sentences)
        
        # 2. Real-time TF-IDF Signal Calculation
        words = text.lower().split()
        l_score = sum(1 for w in words if w in LAYOFF_KEYWORDS)
        a_score = sum(1 for w in words if w in AI_KEYWORDS)

        # 3. Build Input Vector
        input_data = {col: 0.0 for col in feature_cols}
        
        # Fill topic weights
        total = len(sentences) if len(sentences) > 0 else 1
        for t in topics:
            col_name = f"topic_{t}"
            if col_name in input_data:
                input_data[col_name] += (1.0 / total)
        
        # Fill Keyword signals
        if "layoff_signal" in input_data: input_data["layoff_signal"] = float(l_score)
        if "ai_signal" in input_data: input_data["ai_signal"] = float(a_score)
        
        # 4. Inference
        X_in = pd.DataFrame([input_data])[feature_cols]
        prob_up = logit_model.predict_proba(X_in)[0, 1]
        
        # 5. UI Result
        st.metric("Up Probability", f"{prob_up:.2%}")
        
        st.write("### Keyword Signal Strength")
        st.bar_chart({"Layoff": l_score, "AI": a_score})

        st.write("### Topic Distribution:")
        display_cols = []
        for col in feature_cols:
            if col.startswith("topic_"):
                t_id = col.replace("topic_", "")
                display_cols.append(topic_dict.get(t_id, f"Topic {t_id}"))
            else:
                display_cols.append(col)
        
        chart_data = X_in.copy()
        chart_data.columns = display_cols
        non_zero = chart_data.loc[:, (chart_data != 0).any(axis=0)]
        if not non_zero.empty:
            st.bar_chart(non_zero.T)