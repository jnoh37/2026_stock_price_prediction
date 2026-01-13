import json
import re
from pathlib import Path

def clean_and_merge_content(text: str) -> str:
    """
    Removes all newlines and extra spaces to create a continuous block of text.
    """
    # Replace all whitespace (including newlines) with a single space
    merged_text = re.sub(r"\s+", " ", text)
    return merged_text.strip()

def process_conference_file(txt_path: Path) -> dict:
    """
    Parses a single conference call text file.
    Extracts Date and URL, then merges the rest into content.
    """
    raw_text = txt_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    metadata = {"date": None, "url": None}
    content_start_idx = 0

    # Parse metadata from the top of the file
    for i, line in enumerate(lines):
        lower_line = line.lower()
        if lower_line.startswith("date:"):
            metadata["date"] = line.split(":", 1)[1].strip()
        elif lower_line.startswith("url:"):
            metadata["url"] = line.split(":", 1)[1].strip()
        else:
            # First line that isn't Date or URL marks the beginning of content
            content_start_idx = i
            break
    
    # Merge all lines after metadata into a single string without \n
    raw_content = " ".join(lines[content_start_idx:])
    final_content = clean_and_merge_content(raw_content)

    # Structure the data according to requirements
    data = {
        "type": "conference",
        "date": metadata["date"],
        "company_name": txt_path.stem,  # Extracts name from the filename
        "title": None,                 # Always null as requested
        "url": metadata["url"],
        "content": final_content
    }

    return data

def convert_conference_folder(input_dir: str, output_file: str):
    """
    Reads all .txt files from the conference_call directory and saves them as JSONL.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    records = []

    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' not found.")
        return

    # Process all .txt files in the specified directory
    for txt_file in input_path.glob("*.txt"):
        try:
            record = process_conference_file(txt_file)
            records.append(record)
        except Exception as e:
            print(f"Failed to process {txt_file.name}: {e}")

    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL (one JSON object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Processing complete. {len(records)} conference calls saved to {output_file}")

if __name__ == "__main__":
    convert_conference_folder(
        input_dir="data/raw_data/conference_call",
        output_file="data/processed_data/events_conf.jsonl"
    )