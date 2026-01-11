import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 1. Define signal vocabularies
# =========================

LAYOFF_KEYWORDS = [
    "layoff", "layoffs", "job", "jobs", "cut", "cuts",
    "headcount", "workforce", "redundancy",
    "restructuring", "downsizing",
    "cost", "costs", "expense", "expenses",
    "efficiency", "efficiencies"
]

AI_KEYWORDS = [
    "ai", "artificial", "intelligence",
    "automation", "automated",
    "machine", "learning",
    "generative",
    "productivity",
    "digital", "transformation"
]

VOCAB = sorted(set(LAYOFF_KEYWORDS + AI_KEYWORDS))


# =========================
# 2. Load texts from JSONL
# =========================

def load_events(jsonl_path: str):
    texts = []
    meta = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = data.get("content", "")
            if content:
                texts.append(content)
                meta.append({
                    "date": data.get("date"),
                    "title": data.get("title"),
                    "url": data.get("url")
                })

    return texts, meta


# =========================
# 3. Run TF-IDF (keyword-restricted)
# =========================

def run_tfidf(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        vocabulary=VOCAB,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )

    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    return X, feature_names


# =========================
# 4. Build event signal scores
# =========================

def build_signal_scores(X, feature_names):
    layoff_idx = [i for i, f in enumerate(feature_names) if f in LAYOFF_KEYWORDS]
    ai_idx = [i for i, f in enumerate(feature_names) if f in AI_KEYWORDS]

    layoff_score = np.asarray(X[:, layoff_idx].sum(axis=1)).ravel()
    ai_score = np.asarray(X[:, ai_idx].sum(axis=1)).ravel()

    return layoff_score, ai_score


# =========================
# 5. Save results
# =========================

def save_results(meta, layoff_score, ai_score, output_path):
    records = []

    for i in range(len(meta)):
        records.append({
            "date": meta[i]["date"],
            "title": meta[i]["title"],
            "url": meta[i]["url"],
            "layoff_signal": float(layoff_score[i]),
            "ai_signal": float(ai_score[i])
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# 6. Main
# =========================

if __name__ == "__main__":
    INPUT_JSONL = "data/processed_data/events.jsonl"
    OUTPUT_JSONL = "data/processed_data/event_signals.jsonl"

    texts, meta = load_events(INPUT_JSONL)
    X, feature_names = run_tfidf(texts)
    layoff_score, ai_score = build_signal_scores(X, feature_names)

    save_results(meta, layoff_score, ai_score, OUTPUT_JSONL)

    print("Done.")
    print("Documents processed:", len(texts))
    print("Vocabulary used:", feature_names.tolist())
