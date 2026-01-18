import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import os


# =========================
# 1. Company → Ticker mapping
# =========================

COMPANY_TICKER_MAP = {
    "ABN AMRO": "ABN.AS",
    "Barclays": "BARC.L",
    "BNP Paribas": "BNP.PA",
    "Commerzbank": "CBK.DE",
    "Deutsche Bank": "DBK.DE",
    "HSBC": "HSBA.L",
    "ING": "INGA.AS",
    "Julius Baer": "BAER.SW",
    "BankOfAmerica": "BAC",
    "Citigroup": "C",
    "JPMorganChase": "JPM",
    "Visa": "V",
}
# =========================
# 2. Helper Functions
# =========================

def get_price_data(ticker, start_date, end_date):
    """Fetch historical price data from yfinance."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df

def align_to_trading_day(event_date, price_df):
    """Align event date to the nearest future trading day if it falls on a weekend."""
    event_date = pd.to_datetime(event_date)
    trading_days = price_df.index
    if event_date in trading_days:
        return event_date
    future_days = trading_days[trading_days > event_date]
    return future_days[0] if len(future_days) > 0 else None

def compute_returns(event_day, price_df, horizons=[1, 3, 5]):
    """Calculate returns for various time horizons after the event."""
    returns = {}
    for h in horizons:
        try:
            p0 = price_df.loc[event_day]["Close"]
            p1 = price_df.shift(-h).loc[event_day]["Close"]
            returns[f"ret_t_plus_{h}"] = float(p1 / p0 - 1)
        except Exception:
            returns[f"ret_t_plus_{h}"] = None
    return returns

# =========================
# 3. Execution Pipeline
# =========================

if __name__ == "__main__":
    # Load BERTopic sentence-level results
    TOPIC_FILE = "data/processed_data/topic-analysis_results.csv"
    df_sentences = pd.read_csv(TOPIC_FILE)
    
    # Filter for high-confidence topics only (Probability > 0.8)
    df_sentences = df_sentences[df_sentences['probability'] > 0.8].copy()
    
    # Aggregate sentences into Document-level Topic Features
    # This creates a matrix where rows are news articles and columns are topic counts
    topic_features = pd.crosstab(
        [df_sentences['company_name'], df_sentences['date'], df_sentences['title']], 
        df_sentences['topic']
    ).reset_index()

    # Rename numeric topic columns to strings (e.g., 'topic_0', 'topic_1')
    topic_features.columns = [f"topic_{c}" if isinstance(c, (int, np.integer)) else c for c in topic_features.columns]

    results = []

    print("Starting stock data alignment...")
    for idx, row in topic_features.iterrows():
        company = row['company_name']
        ticker = COMPANY_TICKER_MAP.get(company)
        
        if not ticker:
            continue
        
        event_date = pd.to_datetime(row['date'])
        
        # Buffer window to ensure we capture trading days
        start = event_date - timedelta(days=5)
        end = event_date + timedelta(days=15)
        
        price_df = get_price_data(ticker, start, end)
        if price_df.empty:
            continue
            
        aligned_day = align_to_trading_day(event_date, price_df)
        if aligned_day is None:
            continue
            
        # Calculate returns (T+1, T+3, T+5)
        ret = compute_returns(aligned_day, price_df)
        
        # Combine metadata, returns, and topic features
        entry = {
            "date": row['date'],
            "company": company,
            "title": row['title'],
            "ticker": ticker,
            **ret
        }
        
        # Add the fixed topic features
        for col in topic_features.columns:
            if col.startswith("topic_"):
                entry[col] = row[col]
                
        results.append(entry)

    # Save the combined dataset for modeling
    final_df = pd.DataFrame(results)
    output_path = "data/processed_data/topic_returns_combined.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path}. Total events: {len(final_df)}")

# =========================
    # 4. Generate Topic Dictionary (New Section)
    # =========================
    from bertopic import BERTopic

    # Load your trained BERTopic model
    # Ensure the path matches where you saved your model earlier
    topic_model = BERTopic.load("artifacts/bertopic_model")

    # Get topic information (includes Topic ID and representation words)
    topic_info = topic_model.get_topic_info()

    topic_mapping = {}
    for idx, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1: 
            continue # Skip outliers
        
        # Get top 3 keywords for this topic and join them with an underscore
        keywords = "_".join([word for word, prob in topic_model.get_topic(topic_id)[:3]])
        topic_mapping[int(topic_id)] = keywords

    # Save the mapping to a JSON file for the Logistic script
    dict_path = "data/processed_data/topic_dictionary.json"
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(topic_mapping, f, ensure_ascii=False, indent=4)

    print(f"Topic dictionary saved to {dict_path}")