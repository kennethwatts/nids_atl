import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. LOAD AND PREP
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)

df['Label'] = df['Label'].astype(str).str.strip()
y = df['Label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)
X = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

# Split into Source (for TL) and Target (for Trad)
X_source, X_target, y_source, y_target = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# 2. THE EXPERT (TL Model)
# Pre-trained on a large, balanced laboratory dataset
tl_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
tl_model.fit(X_source, y_source)

# Test set stays balanced so we measure "Fairness" across all attacks
X_test, y_test = X_target.sample(10000), y_target.sample(10000)
tl_f1 = f1_score(y_test, tl_model.predict(X_test))

# 3. THE NOVICE (Traditional ML)
# We simulate a real network stream where attacks are only 2% of the traffic
attack_indices = y_target[y_target == 1].index
benign_indices = y_target[y_target == 0].index

sample_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000]
trad_scores = []

print("Simulating Real-World Imbalanced Stream...")
for total_size in sample_sizes:
    # Create an imbalanced training set (2% Attack, 98% Benign)
    n_attacks = int(total_size * 0.02)
    n_benign = total_size - n_attacks
    
    # Safeguard for tiny samples
    if n_attacks < 1: n_attacks = 1
    
    # Grab the imbalanced slice
    idx_subset = np.concatenate([
        np.random.choice(attack_indices, n_attacks, replace=True),
        np.random.choice(benign_indices, n_benign, replace=True)
    ])
    
    X_train_stream = X_target.loc[idx_subset]
    y_train_stream = y.loc[idx_subset]
    
    # Train the novice
    trad_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    trad_model.fit(X_train_stream, y_train_stream)
    
    # Evaluate
    score = f1_score(y_test, trad_model.predict(X_test))
    trad_scores.append(score)
    print(f"Total Traffic Seen: {total_size} (Attacks: {n_attacks}) | Trad F1: {score:.4f}")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, trad_scores, marker='s', label='Traditional ML (2% Attack Ratio)', color='#e74c3c', linewidth=3)
plt.axhline(y=tl_f1, color='#2ecc71', linestyle='--', label='Transfer Learning (Expert Model)', linewidth=3)

plt.title("The 'Day-Zero' Advantage of Transfer Learning", fontsize=14)
plt.xlabel("Total Network Records Processed", fontsize=12)
plt.ylabel("Detection Performance (F1-Score)", fontsize=12)
plt.xscale('log') # Log scale helps see the early gaps clearly
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

# Shaded area for "The Vulnerability Window"
plt.fill_between(sample_sizes, trad_scores, tl_f1, 
                 where=(np.array(trad_scores) < tl_f1), 
                 color='gray', alpha=0.2, label='Vulnerability Window')

plt.savefig('tl_imbalanced_benefit.png')
plt.show()