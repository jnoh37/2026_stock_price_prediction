# 📈 2026_stock_price_prediction

An extensible, interpretable NLP framework for modeling how corporate announcements impact short-term stock price movements. This repository includes a reference end-to-end implementation using layoff and AI-related signals.

This project focuses on how specific announcement themes (e.g. layoffs, AI adoption) are associated with short-term stock price reactions, rather than black-box text embeddings.

---

## 🔍 Project Overview

Corporate announcements often contain signals that influence short-term market reactions.
This project builds an end-to-end, interpretable NLP pipeline that:

* Extracts thematic text signals (Layoff intensity & AI adoption)
* Aligns announcements with actual stock price movements
* Trains a logistic regression model to estimate the probability of price increase
* Provides web interfaces for **data intake**, **prediction**, and **admin-controlled retraining**

Conference call transcripts from major US / EU financial institutions are used to expand the event dataset.

---

## 🧠 Key Idea: Hypothesis-Driven NLP

Instead of asking a model "Is this news good or bad?", we test specific economic hypotheses:

1.  **Efficiency vs. Morale**: Does layoff-related language signal cost-cutting efficiency (positive) or internal instability (negative)?
2.  **Growth Narratives**: Are AI and automation narratives consistently associated with productivity-led growth expectations?

By quantifying these themes via TF-IDF and BERTopic, we convert raw text into actionable "Feature Signals" that explain **why** a prediction was made.

---

## 🧠 Methodology: Hybrid Signals

This project moves beyond opaque "black-box" embeddings by using a hypothesis-driven approach. We combine two transparent signal extraction methods to feed a transparent regression model.

* **Keyword Signals (TF-IDF)**: Quantifies the frequency of specific, pre-defined themes like *Layoffs* and *AI Adoption*.
* **Topic Signals (BERTopic)**: Captures latent semantic contexts through unsupervised clustering of text fragments.
* **Prediction**: A **Logistic Regression** model estimates the probability of a stock price increase on the next trading day ($t+1$).

---

## 🚀 Quick Start

### 1. Installation
Install the required libraries listed in the `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline (Steps 01–06)
Automate the data processing and modeling sequence via the Makefile. This will run the scripts from JSON conversion to the final logistic regression training:
```bash
make all
```

### 3. Launch Web Applications (Steps 07-08)
Each application handles a specific part of the workflow:
```bash
# For real-time signal extraction and predictions
streamlit run script/07_inference_app.py

# For submitting new announcement data to the staging pool
streamlit run script/08_data_intake_app.py
```

## 📁 Project Structure

```
2026_stock_price_prediction/
├── artifacts/                # Model binaries and metadata
│   ├── bertopic_model/       # Pre-trained BERTopic model folder
│   ├── feature_columns.joblib
│   ├── logit_signal_model.joblib
│   └── topic_logit_model.joblib
├── data/
│   ├── pools/                # Staging area for user-submitted events
│   ├── processed_data/       # Structured JSONL and aligned returns data
│   └── raw_data/             # Original TXT news & conference call transcripts
├── script/                   # Full Processing & App pipeline
│   ├── 01_txt_to_json_all.py
│   ├── 02_tfidf_event_signals.py
│   ├── 03_bert_topic.py
│   ├── 04_align_events_multi_company.py
│   ├── 05_train_model.py
│   ├── 06_logistic_regression_event_model.py
│   ├── 07_inference_app.py
│   ├── 08_data_intake_app.py
│   ├── 09_admin_retrain.py
│   └── save_user_upload.py   # Helper for data staging
├── Makefile                  # Orchestrates the core pipeline (01-06)
├── requirements.txt
└── README.md
```
---


## 👤 Authors

Jeeyeon Noh
Jingyi Wang

---

## 📜 Disclaimer

This project is for educational and research purposes only.
It does not constitute financial or investment advice.
