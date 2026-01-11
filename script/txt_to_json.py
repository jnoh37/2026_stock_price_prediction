import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - normalize newlines
    - strip whitespace
    """
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def parse_metadata(lines):
    """
    Parse metadata lines like:
    Source:
    Title:
    Date:
    URL:
    """
    metadata = {
        "source": None,
        "title": None,
        "date": None,
        "url": None
    }

    content_start_idx = 0

    for i, line in enumerate(lines):
        if line.lower().startswith("source:"):
            metadata["source"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("title:"):
            metadata["title"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("date:"):
            metadata["date"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("url:"):
            metadata["url"] = line.split(":", 1)[1].strip()
        else:
            content_start_idx = i
            break

    return metadata, lines[content_start_idx:]


def txt_to_json(txt_path: Path) -> dict:
    """
    Convert a single .txt file into a JSON object.
    """
    raw_text = txt_path.read_text(encoding="utf-8")

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    metadata, content_lines = parse_metadata(lines)

    content = "\n".join(content_lines)

    data = {
        "type": "news",
        "date": metadata["date"],
        "company_name": None,   # intentionally left null for manual mapping
        "title": metadata["title"],
        "url": metadata["url"],
        "content": clean_text(content)
    }

    return data


def convert_folder(input_dir: str, output_file: str):
    """
    Convert all .txt files in a folder into a JSONL file.
    """
    input_dir = Path(input_dir)
    records = []

    for txt_file in input_dir.glob("*.txt"):
        record = txt_to_json(txt_file)
        records.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_folder(
        input_dir="data/raw_data",
        output_file="data/processed_data/events.jsonl"
    )
