import pandas as pd

# 1. DEFINE TARGETS
#input_csv = "cic2018_full_hdr.csv"
#output_csv = "master_stratified_dataset_2018.csv"
input_csv = "cic2017_full_hdr.csv"
output_csv = "master_stratified_dataset_2017.csv"

# Quotas
RARE_THRESHOLD = 2000    # If a label has fewer than this, take 100% of them
COMMON_CAP = 50000       # Cap large attack categories at 50k
BENIGN_CAP = 250000      # Cap Benign at 250k to maintain a 1:5 ratio roughly

print("Starting Rare-Attack-Hunter extraction...")

chunks = pd.read_csv(input_csv, chunksize=500000)
storage = []
counts = {}

for i, chunk in enumerate(chunks):
    chunk['Label'] = chunk['Label'].astype(str).str.strip()
    
    for label in chunk['Label'].unique():
        if label not in counts: counts[label] = 0
        
        # Determine the dynamic cap for this specific label
        if label.lower() == 'benign':
            current_cap = BENIGN_CAP
        else:
            # We don't know if it's rare or common yet, so we 
            # set it to the COMMON_CAP. If it ends up being rare, 
            # we'll naturally just get all of them.
            current_cap = COMMON_CAP
            
        if counts[label] < current_cap:
            needed = current_cap - counts[label]
            data = chunk[chunk['Label'] == label].head(needed)
            
            storage.append(data)
            counts[label] += len(data)
            
    if i % 5 == 0: print(f"Processed {i * 0.5}M rows... Tally: {counts}")

# 2. COMBINE & SHUFFLE
print("Combining fragments and applying final shuffle...")
final_df = pd.concat(storage).sample(frac=1, random_state=42).reset_index(drop=True)

# 3. SAVE
final_df.to_csv(output_csv, index=False)
print(f"Success! Master Dataset saved with {len(final_df)} records.")
print("\n--- FINAL CLASS DISTRIBUTION ---")
print(final_df['Label'].value_counts())