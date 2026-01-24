import json
import re
import os
import pandas as pd
import numpy as np
from bertopic import BERTopic

# 1. Path Configuration
input_path = 'data/processed_data/events_all.jsonl'
output_dir = 'data/processed_data'
model_dir = 'artifacts/bertopic_model'

os.makedirs(output_dir, exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

# 2. Load the .jsonl file
raw_data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

# 3. Process documents into sentences while keeping metadata
sentence_data = []
metadata_list = []

for item in raw_data:
    content = item.get("content", "")
    # Split text by sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    for sentence in sentences:
        clean_sent = sentence.strip()
        if len(clean_sent) > 20:
            sentence_data.append(clean_sent)
            metadata_list.append({
                "company_name": item.get("company_name"),
                "date": item.get("date"),
                "title": item.get("title")
            })

# 4. Train BERTopic
topic_model = BERTopic(calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(sentence_data)

# 5. Save the model using BERTopic's native save method
topic_model.save(
    model_dir, 
    serialization="safetensors", 
    save_ctfidf=True, 
    save_embedding_model=True
)
print(f"Model saved using BERTopic native method to: {model_dir}")

# 6. Create and save the results DataFrame
results_df = pd.DataFrame(metadata_list)
results_df['sentence'] = sentence_data
results_df['topic'] = topics

# Extract probability for the assigned topic
results_df['probability'] = [np.max(p) if t != -1 else 0 for t, p in zip(topics, probs)]

# Save results to CSV
csv_save_path = os.path.join(output_dir, 'topic-analysis_results.csv')
results_df.to_csv(csv_save_path, index=False)
print(f"Results saved to: {csv_save_path}")