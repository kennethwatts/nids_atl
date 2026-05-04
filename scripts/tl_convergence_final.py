import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. DATA PREP
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
df['Label'] = df['Label'].astype(str).str.strip()
y = df['Label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)
X = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

X_source, X_target, y_source, y_target = train_test_split(X, y, test_size=0.6, stratify=y, random_state=42)

# 2. MODELS
# Static TL (Pre-trained on 2017/Source)
static_tl = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_source, y_source)

# Shared Test Set
X_test, y_test = X_target.sample(10000, random_state=42), y_target.sample(10000, random_state=42)
static_f1 = f1_score(y_test, static_tl.predict(X_test))

# 3. CONVERGENCE SIMULATION
attack_indices = y_target[y_target == 1].index
benign_indices = y_target[y_target == 0].index

# Extending samples out to 50k to find convergence
sample_sizes = [100, 1000, 5000, 10000, 20000, 35000, 50000]
trad_scores = []
adaptive_scores = []

print("Running deep convergence simulation...")
for total_size in sample_sizes:
    # 2% Imbalanced Stream
    n_attacks = max(1, int(total_size * 0.02))
    n_benign = total_size - n_attacks
    
    idx_subset = np.concatenate([
        np.random.choice(attack_indices, n_attacks, replace=True),
        np.random.choice(benign_indices, n_benign, replace=True)
    ])
    X_train_stream, y_train_stream = X_target.loc[idx_subset], y.loc[idx_subset]
    
    # TRADITIONAL ML
    trad_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_stream, y_train_stream)
    trad_scores.append(f1_score(y_test, trad_model.predict(X_test)))
    
    # ADAPTIVE TL (Fine-tuned on the stream)
    # We simulate adaptation by combining source knowledge with new stream data
    X_adaptive = pd.concat([X_source.sample(2000), X_train_stream])
    y_adaptive = pd.concat([y_source.sample(2000), y_train_stream])
    adaptive_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_adaptive, y_adaptive)
    adaptive_scores.append(f1_score(y_test, adaptive_model.predict(X_test)))
    
    print(f"Total: {total_size} | Trad: {trad_scores[-1]:.3f} | Adaptive: {adaptive_scores[-1]:.3f}")

# 4. PLOT
plt.figure(figsize=(12, 7))
plt.plot(sample_sizes, trad_scores, 'r-o', label='Traditional ML (Novice)', linewidth=2)
plt.plot(sample_sizes, adaptive_scores, 'b-s', label='Adaptive TL (Hybrid)', linewidth=2)
plt.axhline(y=static_f1, color='g', linestyle='--', label='Static TL (Expert)', linewidth=2)

plt.xscale('log')
plt.title("Convergence Analysis: The Enduring Value of Transfer Learning", fontsize=14)
plt.xlabel("Total Network Traffic Records (Log Scale)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc='lower right')

# Shade the "Vulnerability Window" where TL is strictly necessary
plt.fill_between(sample_sizes, trad_scores, static_f1, 
                 where=(np.array(trad_scores) < static_f1), 
                 color='orange', alpha=0.1, label='Vulnerability Window')

plt.savefig('tl_convergence_final.png')
plt.show()