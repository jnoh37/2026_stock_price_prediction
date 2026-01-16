import json
import uuid
from datetime import datetime
from pathlib import Path

POOL_PATH = Path("data/pools/user_upload_pool.jsonl")
POOL_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_user_event(
    raw_text: str,
    layoff_signal: float,
    ai_signal: float,
    source: str = "streamlit"
):
    """
    Save user-submitted event into a staging pool.
    This data is NOT used for training unless manually approved.
    """

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "source": source,
        "content": raw_text,
        "layoff_signal": layoff_signal,
        "ai_signal": ai_signal,
        "approved_for_training": False  # key control flag
    }

    with open(POOL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record["id"]
