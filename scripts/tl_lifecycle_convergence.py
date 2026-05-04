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

# Simulate Domain Shift: Source (Knowledge) vs Target (Live Environment)
X_source, X_target, y_source, y_target = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

# 2. CONFIGURATION
iterations = 30 # Reduced for speed over large sample sizes
sample_sizes = [100, 1000, 5000, 10000, 25000, 50000, 100000, 200000]
trad_results = np.zeros((iterations, len(sample_sizes)))
adapt_results = np.zeros((iterations, len(sample_sizes)))

print(f"Starting Monte Carlo analysis up to {max(sample_sizes)} samples...")

for i in range(iterations):
    # Static Reference (The 2017 Knowledge)
    X_test = X_target.sample(10000)
    y_test = y_target.loc[X_test.index]
    
    for j, size in enumerate(sample_sizes):
        # 1% Imbalanced Stream (More realistic rarity)
        n_attacks = max(1, int(size * 0.01))
        idx = np.concatenate([
            np.random.choice(y_target[y_target==1].index, n_attacks, replace=True),
            np.random.choice(y_target[y_target==0].index, size - n_attacks, replace=True)
        ])
        X_stream, y_stream = X_target.loc[idx], y_target.loc[idx]
        
        # Traditional ML
        m_trad = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X_stream, y_stream)
        trad_results[i, j] = f1_score(y_test, m_trad.predict(X_test))
        
        # Adaptive TL (Combines Source Knowledge with Stream Data)
        X_adapt = pd.concat([X_source.sample(1000), X_stream])
        y_adapt = pd.concat([y_source.sample(1000), y_stream])
        m_adapt = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X_adapt, y_adapt)
        adapt_results[i, j] = f1_score(y_test, m_adapt.predict(X_test))
        
    if (i+1) % 5 == 0: print(f"Completed {i+1} iterations...")

# 3. METRICS CALCULATION
trad_mean = np.mean(trad_results, axis=0)
adapt_mean = np.mean(adapt_results, axis=0)

# Calculate the "Transfer Advantage" (AUC Gap)
# This is essentially the integral of the performance difference
advantage_area = np.trapz(adapt_mean - trad_mean, x=np.log10(sample_sizes))

# 4. PLOTTING
plt.figure(figsize=(12, 7))
plt.plot(sample_sizes, trad_mean, 'r-o', label='Traditional ML', linewidth=2)
plt.plot(sample_sizes, adapt_mean, 'b-s', label='Adaptive TL', linewidth=2)

plt.xscale('log')
plt.title(f"Performance Life-cycle: TL Advantage Area = {advantage_area:.2f}", fontsize=14)
plt.xlabel("Records Processed (Log Scale)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)

# Shade the Cumulative Advantage Area
plt.fill_between(sample_sizes, trad_mean, adapt_mean, color='blue', alpha=0.15, label='TL Cumulative Advantage')

plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig('tl_lifecycle_convergence.png')
plt.show()

print(f"\nConvergence Analysis Complete.")
print(f"Total Transfer Advantage (Area): {advantage_area:.4f}")