# 📈 2026_stock_price_prediction

An end-to-end, interpretable NLP system for modeling how corporate announcements impact short-term stock price movements.

This project focuses on **hypothesis-driven text signals** (e.g. layoffs, AI adoption) rather than black-box embeddings, and provides a fully reproducible pipeline from raw text to prediction, including web-based inference and retraining.

---

## 🚀 What This Project Does (TL;DR)

**Input**  
Corporate announcements (news articles, conference call transcripts)

**Process**  
→ Text signal extraction (TF-IDF + BERTopic)  
→ Event–stock alignment  
→ Interpretable logistic regression modeling  

**Output**  
→ Probability of next-day stock price increase  
→ Transparent feature-level explanations  
→ Web apps for inference, data intake, and admin retraining  

---

## 🧪 Execution Environment (Databricks)

This project has been **fully reproduced and validated in a Databricks Workspace** environment.

- All core data processing and modeling steps are runnable as **Databricks notebooks**
- Folder structure mirrors the local pipeline (`raw_data → processed_data → artifacts`)
- The workflow supports collaborative experimentation and scalable execution
- Results are consistent between **local execution** and **Databricks Workspace** runs

This makes the project suitable for both **local research workflows** and **cloud-based analytics / enterprise data platforms**.

---

## 🔍 Project Overview

Corporate announcements often contain signals that influence short-term market reactions.  
This project builds an **end-to-end, interpretable NLP pipeline** that:

- Extracts thematic text signals (e.g. Layoff intensity & AI adoption)  
- Aligns announcement events with observed stock price movements  
- Trains a logistic regression model to estimate price movement probability  
- Provides web interfaces for:
  - Real-time prediction
  - Data intake
  - Admin-controlled retraining

Conference call transcripts from major US / EU financial institutions are used to expand the event dataset.

---

## 🧠 Key Idea: Hypothesis-Driven NLP

Instead of asking *“Is this news positive or negative?”*, this project tests **explicit economic hypotheses**:

1. **Efficiency vs. Morale**  
   Does layoff-related language indicate cost efficiency (positive) or organizational instability (negative)?

2. **Growth Narratives**  
   Are AI and automation narratives systematically associated with productivity-led growth expectations?

By quantifying these themes via **TF-IDF** and **BERTopic**, raw text is converted into **interpretable feature signals** that explain *why* a prediction is made.

---

## 🧠 Methodology: Hybrid, Interpretable Signals

This project deliberately avoids opaque embeddings in favor of transparent signals:

- **Keyword Signals (TF-IDF)**  
  Measures the intensity of predefined themes such as *Layoffs* and *AI Adoption*.

- **Topic Signals (BERTopic)**  
  Captures latent semantic contexts through unsupervised topic modeling.

- **Prediction Model**  
  A **Logistic Regression** model estimates the probability of a stock price increase on day *t+1*.

---

## ⚙️ Pipeline Overview

The core modeling pipeline is automated via `Makefile` and consists of six steps:

| Step | Description |
|-----:|-------------|
| 01 | Convert raw TXT announcements to structured JSON |
| 02 | Extract TF-IDF-based event signals |
| 03 | Train BERTopic model and generate topic signals |
| 04 | Align events with multi-company stock returns |
| 05 | Train feature-based prediction model |
| 06 | Train final logistic regression event model |

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Steps 01–06)
```bash
make all
```

### 3. Launch Web Applications

```bash
# Real-time inference & prediction
streamlit run script/07_inference_app.py

# Submit new announcements to the staging pool
streamlit run script/08_data_intake_app.py
```

Admin retraining can be triggered via:
```bash
streamlit run script/09_admin_retrain.py
```

---

## 📁 Project Structure

```
2026_stock_price_prediction/
├── artifacts/                # Trained models and feature metadata
│   ├── bertopic_model/
│   ├── feature_columns.joblib
│   ├── logit_signal_model.joblib
│   └── topic_logit_model.joblib
├── data/
│   ├── raw_data/             # News & conference call transcripts
│   ├── processed_data/       # JSONL signals & aligned returns
│   └── pools/                # User-submitted event staging area
├── script/
│   ├── 01_txt_to_json_all.py
│   ├── 02_tfidf_event_signals.py
│   ├── 03_bert_topic.py
│   ├── 04_align_events_multi_company.py
│   ├── 05_train_model.py
│   ├── 06_logistic_regression_event_model.py
│   ├── 07_inference_app.py
│   ├── 08_data_intake_app.py
│   ├── 09_admin_retrain.py
│   └── save_user_upload.py
├── Makefile
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