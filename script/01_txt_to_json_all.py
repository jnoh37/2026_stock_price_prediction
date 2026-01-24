import os
import json

def parse_txt_to_json(file_path):
    """
    Parses a single text file into a structured dictionary.
    Identifies metadata based on 'Key: Value' format and treats the rest as content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = {}
    content_start_idx = 0
    
    # Assign type based on the folder directory name
    if 'conference_call' in file_path:
        data['type'] = 'conference'
    elif 'news' in file_path:
        data['type'] = 'news'
    else:
        data['type'] = 'unknown'

    # Extract metadata from the top of the file
    for i, line in enumerate(lines):
        # Checking for 'Key: Value' pattern (limit to first 10 lines)
        if ':' in line and i < 10:
            key_part, value_part = line.split(':', 1)
            key = key_part.strip().lower()
            value = value_part.strip()
            
            # Mapping relevant file keys to JSON fields
            if key == 'date':
                data['date'] = value
            elif key == 'url':
                data['url'] = value
            elif key == 'name':
                data['company_name'] = value
            elif key == 'source':
                data['source'] = value
            elif key == 'title':
                data['title'] = value
            # 'quarter' is intentionally ignored here
            
            content_start_idx = i + 1
        else:
            # Stop parsing metadata if the line doesn't follow 'Key: Value' format
            break

    # Consolidate the remaining lines into the 'content' field
    data['content'] = " ".join([l.strip() for l in lines[content_start_idx:] if l.strip()])
    
    # Ensure 'title' exists in the dictionary, default to null (None) if missing
    if 'title' not in data:
        data['title'] = None
        
    return data

def process_and_save_jsonl(input_base_path, output_file_path):
    """
    Traverses the directory and saves all parsed data into a single .jsonl file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        count = 0
        for root, dirs, files in os.walk(input_base_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    parsed_data = parse_txt_to_json(file_path)
                    
                    # Write each dictionary as a single JSON line
                    f_out.write(json.dumps(parsed_data, ensure_ascii=False) + '\n')
                    count += 1
        return count

# Main Execution
if __name__ == "__main__":
    # Input and Output paths
    input_dir = 'data/raw_data'
    output_path = 'data/processed_data/events_all.jsonl'
    
    total_processed = process_and_save_jsonl(input_dir, output_path)
    print(f"Successfully processed {total_processed} files.")
    print(f"Output saved to: {output_path}")