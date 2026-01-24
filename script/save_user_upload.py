import json
import uuid
from datetime import datetime
from pathlib import Path

# Path for the data pool
POOL_PATH = Path("data/pools/user_upload_pool.jsonl")
POOL_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_user_event(raw_text, combined_signals, source="streamlit"):
    """
    Save user-submitted event into a staging pool.
    This data can be used for future model re-training.
    """
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "source": source,
        "content": raw_text,
        "signals": combined_signals,
        "approved_for_training": False
    }

    with open(POOL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record["id"]