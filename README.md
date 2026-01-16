# 2026_stock_price_prediction

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

## 🧠 Key Idea

Instead of using opaque embeddings, this project adopts a **hypothesis-driven NLP approach**:

* Layoff-related language → often associated with short-term negative sentiment
* AI / automation narratives → often associated with productivity and growth expectations

These signals are quantified using TF-IDF–based keyword scoring and fed into a transparent regression model.

---

## 📁 Project Structure

```
2026_stock_price_prediction/
│
├── artifacts/
│   └── logit_signal_model.joblib      # Trained logistic regression model
│
├── data/
│   ├── raw_data/                      # Raw news / announcement text files (.txt)
│   │   └── conference_call/           # Conference call transcripts
│   ├── processed_data/
│   │   ├── events.jsonl
│   │   ├── events_conf.jsonl
│   │   ├── event_signals.jsonl
│   │   └── event_returns_multi_company.csv
│   └── pools/                         # User-submitted events (not auto-used)
│
├── script/
│   ├── txt_to_json.py                 # TXT → JSON conversion
│   ├── txt_to_json_conf.py            # Conference call TXT → JSONL conversion
│   ├── tfidf_event_signals.py         # TF-IDF signal extraction
│   ├── align_events_multi_company.py  # Event–price alignment
│   ├── train_model.py                 # Model training
│   ├── 01_data_intake_app.py           # Streamlit: user data submission
│   ├── 02_inference_app.py             # Streamlit: prediction only
│   └── 03_admin_retrain.py             # Streamlit: admin retraining
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Methodology

### 1️⃣ Text Processing

* Raw `.txt` announcements parsed into structured JSON
* Metadata (date, title, URL) + cleaned content
* Conference calls assigned synthetic titles (e.g. `{company}_conference_call`)

### 2️⃣ Signal Extraction

* TF-IDF–based keyword scoring
* Two interpretable signals:

  * `layoff_signal`
  * `ai_signal`

### 3️⃣ Event Study

* Events aligned to the next trading day using Yahoo Finance data
* Short-term returns computed (t+1, t+3, t+5)

### 4️⃣ Modeling

* Logistic Regression
* Inputs: layoff_signal, ai_signal
* Output: Probability of stock price increase

---

## 🌐 Web Applications (Streamlit)

This project deliberately separates **data submission**, **prediction**, and **model retraining**.

### 1️⃣ Data Intake App (User-facing)

**File:** `script/01_data_intake_app.py`

* Users submit new announcements or excerpts
* Signals are computed and stored in `data/pools/`
* Data is **not** automatically used for training

Run:

```
streamlit run script/01_data_intake_app.py
```

---

### 2️⃣ Inference App (Prediction only)

**File:** `script/02_inference_app.py`

* Loads pre-trained model from `/artifacts`
* Runs real-time up/down probability prediction
* Does **not** save user inputs

Run:

```
streamlit run script/02_inference_app.py
```

---

### 3️⃣ Admin Retrain App (Human-in-the-loop)

**File:** `script/03_admin_retrain.py`

* Reviews pooled user events
* Allows manual inclusion/exclusion
* Triggers retraining explicitly

Design principle:

> **No automatic learning without human approval**

Run:

```
streamlit run script/03_admin_retrain.py
```

---

## 📦 Dependencies

See `requirements.txt`. Core libraries include:

* scikit-learn
* pandas
* numpy
* yfinance
* streamlit

---

## ⚠️ Limitations

* Small event sample size
* Market-wide movements not fully controlled
* Designed for interpretability & research, not trading execution

---

## 🚀 Future Improvements

* Expand labeled event dataset
* Add market-adjusted abnormal returns
* Include sector / index controls
* Compare against embedding-based models (e.g. BERT)

---

## 👤 Authors

Jeeyeon Noh
Jingyi Wang

---

## 📜 Disclaimer

This project is for educational and research purposes only.
It does not constitute financial or investment advice.
