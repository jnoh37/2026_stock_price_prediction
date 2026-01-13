import json
import yfinance as yf
import pandas as pd
from datetime import timedelta


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


def infer_ticker_from_title(title: str):
    """
    Infer ticker from event title using rule-based matching.
    """
    title_lower = title.lower()
    for company, ticker in COMPANY_TICKER_MAP.items():
        if company.lower() in title_lower:
            return company, ticker
    return None, None


# =========================
# 2. Load event signals
# =========================

def load_event_signals(jsonl_path):
    events = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    return events


# =========================
# 3. Get stock prices
# =========================

def get_price_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )
    return df


# =========================
# 4. Align event date to trading day
# =========================

def align_to_trading_day(event_date, price_df):
    event_date = pd.to_datetime(event_date)
    trading_days = price_df.index

    if event_date in trading_days:
        return event_date

    future_days = trading_days[trading_days > event_date]
    if len(future_days) == 0:
        return None

    return future_days[0]


# =========================
# 5. Compute event returns
# =========================

def compute_returns(event_day, price_df, horizons=[1, 3, 5]):
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
# 6. Main pipeline
# =========================

if __name__ == "__main__":

    EVENT_FILE = "data/processed_data/event_signals.jsonl"

    events = load_event_signals(EVENT_FILE)
    results = []

    for e in events:
        if not e.get("date") or not e.get("title"):
            continue

        company, ticker = infer_ticker_from_title(e["title"])
        if ticker is None:
            continue  # skip events we cannot map safely

        event_date = pd.to_datetime(e["date"])

        # Fetch price window (buffer around event)
        start = event_date - timedelta(days=5)
        end = event_date + timedelta(days=10)

        price_df = get_price_data(ticker, start, end)
        if price_df.empty:
            continue

        aligned_day = align_to_trading_day(event_date, price_df)
        if aligned_day is None:
            continue

        ret = compute_returns(aligned_day, price_df)

        results.append({
            "event_date": e["date"],
            "trading_date": str(aligned_day.date()),
            "company": company,
            "ticker": ticker,
            "title": e["title"],
            "layoff_signal": e["layoff_signal"],
            "ai_signal": e["ai_signal"],
            **ret
        })

    df = pd.DataFrame(results)
    print(df)

    # Optional: save result
    df.to_csv("data/processed_data/event_returns_multi_company.csv", index=False)
