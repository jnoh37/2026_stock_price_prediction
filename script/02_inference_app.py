import streamlit as st
import joblib
import numpy as np
import pandas as pd
from bertopic import BERTopic
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BERTOPIC_PATH = PROJECT_ROOT / "artifacts" / "bertopic_model"
DICT_PATH = PROJECT_ROOT / "data" / "processed_data" / "topic_dictionary.json"
LOGIT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "topic_logit_model.joblib"
FEATURE_COLS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.joblib"

@st.cache_resource
def load_artifacts():
    topic_model = BERTopic.load(str(BERTOPIC_PATH))
    logit_model = joblib.load(LOGIT_MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH) # Important: match the training column order

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        topic_dict = json.load(f)
    return topic_model, logit_model, feature_cols, topic_dict

st.title("Stock Price Prediction")
topic_model, logit_model, feature_cols, topic_dict = load_artifacts()

text = st.text_area("Enter news text for prediction:", height=260)

if st.button("Predict!"):
    if text.strip():
        # 1. Split into sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
        
        # 2. Extract Topic distribution
        topics, _ = topic_model.transform(sentences)
        
        # 3. Align with Training Features (Important!)
        # Create a zero-vector for all possible topics
        input_data = {col: 0.0 for col in feature_cols}
        
        # Fill in the topics found in the user input
        total = len(sentences)
        for t in topics:
            col_name = f"topic_{t}"
            if col_name in input_data:
                input_data[col_name] += (1.0 / total)
        
        # 4. Inference
        X_in = pd.DataFrame([input_data])[feature_cols]
        display_cols = []
        for col in feature_cols:
            t_id = col.replace("topic_", "")
            friendly_name = topic_dict.get(t_id, f"Topic {t_id}")
            display_cols.append(friendly_name)
        chart_data = X_in.copy()
        chart_data.columns = display_cols
            
        prob_up = logit_model.predict_proba(X_in)[0, 1]
        
        st.metric("Up Probability", f"{prob_up:.2%}")
        st.write("### Topic Distribution in this text:")
        non_zero_topics = chart_data.loc[:, (chart_data != 0).any(axis=0)]
        if not non_zero_topics.empty:
            st.bar_chart(non_zero_topics.T)
        else:
            st.info("No recognizable topics found in the input text.")