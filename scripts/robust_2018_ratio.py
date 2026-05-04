import pandas as pd

# Load the robust 2018 dataset
df = pd.read_csv('robust_2018_final.csv')

# Count the occurrences of each label
# (Assuming 0 is Benign and 1 is Attack based on your previous methodology)
counts = df['Label'].value_counts()
total = len(df)

benign_count = counts.get(0, 0)
attack_count = counts.get(1, 0)

# Calculate percentages
benign_pct = (benign_count / total) * 100
attack_pct = (attack_count / total) * 100

print(f"Total Samples: {total}")
print(f"Benign Flows (0): {benign_count} ({benign_pct:.2f}%)")
print(f"Attack Flows (1): {attack_count} ({attack_pct:.2f}%)")
print(f"Ratio (Benign:Attack): {benign_count/attack_count:.1f}:1")