import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)

# Clean and Binary Labeling
y = df['Label'].astype(str).str.strip().apply(lambda x: 0 if x.lower() == 'benign' else 1)
X = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

# Split for testing
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 2. THE SCENARIO
# We simulate a "TL Model" by training it on a completely different slice 
# (simulating a 2017 pre-trained model)
X_source, _, y_source, _ = train_test_split(X_train_full, y_train_full, train_size=10000, stratify=y_train_full, random_state=7)
tl_model = RandomForestClassifier(n_estimators=100, random_state=42)
tl_model.fit(X_source, y_source)
tl_fixed_score = f1_score(y_test, tl_model.predict(X_test))

# 3. THE "COLD START" (Traditional ML)
sample_sizes = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000]
trad_scores = []

print("Running Cold-Start simulation...")
for size in sample_sizes:
    # Take a tiny slice of 'new' data
    X_small = X_train_full.head(size)
    y_small = y_train_full.head(size)
    
    trad_model = RandomForestClassifier(n_estimators=100, random_state=42)
    trad_model.fit(X_small, y_small)
    
    score = f1_score(y_test, trad_model.predict(X_test))
    trad_scores.append(score)
    print(f"Samples: {size} | Traditional F1: {score:.4f}")

# 4. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, trad_scores, marker='o', label='Traditional ML (Learning from scratch)', color='red', linewidth=2)
plt.axhline(y=tl_fixed_score, color='green', linestyle='--', label='Transfer Learning (Pre-trained)', linewidth=2)

# Annotate the crossover point
plt.annotate('Crossover Point', xy=(1000, tl_fixed_score), xytext=(1200, tl_fixed_score-0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title("The Value of Transfer Learning: Cold Start vs. Pre-trained", fontsize=14)
plt.xlabel("Number of Samples in New Environment", fontsize=12)
plt.ylabel("Detection Performance (F1-Score)", fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('tl_vs_traditional_learning_curve.png')
plt.show()