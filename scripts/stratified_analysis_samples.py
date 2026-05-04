import pandas as pd
import numpy as np

# 1. SETTINGS
full_dataset_path = "cic2018_full_working_file.csv" # Change to your large file path
output_path = "stratified_analysis_samples.csv"
samples_per_class = 5000 # Targets 5k of each, or max available

# 2. STRATIFIED EXTRACTION
print("Scanning large dataset for balanced samples...")
chunks = pd.read_csv(full_dataset_path, chunksize=100000)
sampled_chunks = []

for i, chunk in enumerate(chunks):
    chunk['Label'] = chunk['Label'].astype(str).str.strip()
    
    # Take a fraction of each class from this chunk
    for label in chunk['Label'].unique():
        label_data = chunk[chunk['Label'] == label]
        # Randomly sample if chunk has too many, otherwise take all available
        n_to_take = min(len(label_data), 500) 
        sampled_chunks.append(label_data.sample(n=n_to_take, random_state=42))
    
    if i % 10 == 0: print(f"Processed {i*100000} rows...")

# 3. COMBINE AND SAVE
final_df = pd.concat(sampled_chunks)
# Cap it to our total target per class to keep the file size manageable
final_df = final_df.groupby('Label').apply(lambda x: x.sample(n=min(len(x), samples_per_class))).reset_index(drop=True)

final_df.to_csv(output_path, index=False)
print(f"Success! Saved balanced dataset to {output_path}")
print(final_df['Label'].value_counts())