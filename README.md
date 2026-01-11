# 2026_stock_price_prediction

An extensible, interpretable NLP framework for modeling how corporate announcements impact short-term stock price movements. This repository includes a reference end-to-end implementation using layoff and AI-related signals.

This project focuses on how specific announcement themes (e.g. layoffs, AI adoption) are associated with short-term stock price reactions, rather than black-box text embeddings.

# 🔍 Project Overview

Corporate announcements often contain signals that influence short-term market reactions.
This project builds an end-to-end, interpretable NLP pipeline that:

Extracts thematic text signals (Layoff intensity & AI adoption)

Aligns announcements with actual stock price movements

Trains a logistic regression model to estimate the probability of price increase

Provides a web interface for real-time text-based inference

# 🧠 Key Idea

Instead of using opaque embeddings, this project adopts a hypothesis-driven NLP approach:

Layoff-related language → often associated with short-term negative sentiment

AI / automation narratives → often associated with productivity and growth expectations

These signals are quantified using TF-IDF–based keyword scoring and fed into a transparent regression model.

# 📁 Project Structure

2026_stock_price_prediction/
│
├── artifacts/
│   └── logit_signal_model.joblib      # Trained logistic regression model
│
├── data/
│   ├── raw_data/                      # Raw news / announcement text files (.txt)
│   ├── processed_data/
│   │   ├── events.jsonl               # Structured events
│   │   ├── event_signals.jsonl        # Extracted NLP signals
│   │   └── event_returns_multi_company.csv
│
├── script/
│   ├── txt_to_json.py                 # TXT → JSON conversion
│   ├── tfidf_event_signals.py         # TF-IDF signal extraction
│   ├── align_events_multi_company.py  # Event–price alignment
│   ├── train_model.py                 # Model training
│   ├── logistic_regression_event_model.py
│   └── app.py                         # Streamlit web app
│
├── README.md
├── requirements.txt
└── .gitignore

# ⚙️ Methodology
1️⃣ Text Processing

Raw .txt announcements are parsed into structured JSON

Metadata (date, title, URL) + cleaned content

2️⃣ Signal Extraction

TF-IDF–based keyword scoring

Two interpretable signals:

layoff_signal

ai_signal

3️⃣ Event Study

Events aligned to the next trading day using Yahoo Finance data

Short-term returns computed (t+1, t+3, t+5)

4️⃣ Modeling

Logistic Regression

Inputs: layoff_signal, ai_signal

Output: Probability of stock price increase

# 📈 Example Output
{
  "layoff_signal": 2.21,
  "ai_signal": 0.32,
  "up_probability": 0.38,
  "down_probability": 0.62
}


Interpretation:
High layoff intensity and limited AI-related language are associated with a lower probability of short-term price increase.

# 🌐 Web Demo

The project includes a Streamlit web application that allows users to:

Paste a news article or announcement

Instantly obtain:

Layoff & AI signal values

Up / Down probability

Model explanation notes

Run locally
pip install -r requirements.txt
streamlit run script/app.py

# 📦 Dependencies

See requirements.txt. Core libraries include:

scikit-learn

pandas

numpy

yfinance

streamlit

# ⚠️ Limitations

Small event sample size

Market-wide movements not fully controlled

Designed for interpretability & research, not trading execution

# 🚀 Future Improvements

Expand event dataset

Add market-adjusted abnormal returns

Include sector or index controls

Compare with embedding-based models (e.g. BERT)

# 👤 Author

Developed as an interpretable NLP + finance research project.

# 📜 Disclaimer

This project is for educational and research purposes only.
It does not constitute financial or investment advice.
