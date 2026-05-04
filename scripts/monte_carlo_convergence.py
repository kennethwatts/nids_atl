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

# Simulate 2017 (Source) and 2018 (Target)
X_source, X_target, y_source, y_target = train_test_split(X, y, test_size=0.6, stratify=y, random_state=42)

# 2. MONTE CARLO CONFIG
iterations = 50
sample_sizes = [100, 1000, 5000, 10000, 20000, 50000]
trad_results = np.zeros((iterations, len(sample_sizes)))
adaptive_results = np.zeros((iterations, len(sample_sizes)))

print(f"Starting {iterations}-run Monte Carlo simulation...")

for i in range(iterations):
    # Pre-train Static Expert for this iteration
    static_tl = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X_source.sample(5000), y_source.sample(5000))
    X_test = X_target.sample(5000)
    y_test = y_target.loc[X_test.index]
    
    for j, size in enumerate(sample_sizes):
        # 2% Attack Stream
        n_attacks = max(1, int(size * 0.02))
        idx = np.concatenate([
            np.random.choice(y_target[y_target==1].index, n_attacks),
            np.random.choice(y_target[y_target==0].index, size - n_attacks)
        ])
        X_stream, y_stream = X_target.loc[idx], y_target.loc[idx]
        
        # Traditional
        m_trad = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X_stream, y_stream)
        trad_results[i, j] = f1_score(y_test, m_trad.predict(X_test))
        
        # Adaptive (Mix of source and stream)
        X_adapt = pd.concat([X_source.sample(1000), X_stream])
        y_adapt = pd.concat([y_source.sample(1000), y_stream])
        m_adapt = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X_adapt, y_adapt)
        adaptive_results[i, j] = f1_score(y_test, m_adapt.predict(X_test))
        
    if (i+1) % 10 == 0: print(f"Completed {i+1} iterations...")

# 3. STATISTICAL AGGREGATION
trad_mean = np.mean(trad_results, axis=0)
trad_std = np.std(trad_results, axis=0)
adapt_mean = np.mean(adaptive_results, axis=0)
adapt_std = np.std(adaptive_results, axis=0)

# 4. PLOT
plt.figure(figsize=(12, 7))
plt.plot(sample_sizes, trad_mean, 'r-o', label='Traditional ML (Novice)', linewidth=2)
plt.fill_between(sample_sizes, trad_mean-trad_std, trad_mean+trad_std, color='red', alpha=0.1)

plt.plot(sample_sizes, adapt_mean, 'b-s', label='Adaptive TL (Hybrid)', linewidth=2)
plt.fill_between(sample_sizes, adapt_mean-adapt_std, adapt_mean+adapt_std, color='blue', alpha=0.1)

plt.xscale('log')
plt.title("Monte Carlo Convergence Analysis (50 Iterations)", fontsize=14)
plt.xlabel("Records Processed (Log Scale)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig('monte_carlo_convergence.png')
plt.show()