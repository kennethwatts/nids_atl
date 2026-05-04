import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# 1. LOAD AND PREP
file_path = "master_stratified_dataset.csv"
# utf-8-sig handles that BOM (\ufeff) error we saw earlier
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)

# Clean and Binary Labeling (Attack vs. Benign)
df['Label'] = df['Label'].astype(str).str.strip()
y = df['Label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)

# Force Numeric and Clean
X = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

# 2. TRAIN/TEST SPLIT
# Using a 50k/50k split for statistical significance and speed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=50000, test_size=50000, stratify=y, random_state=42
)

# Standardize for SVM, QDA, and NB
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 3. DEFINE MODELS
models = {
    "DT": DecisionTreeClassifier(max_depth=10),
    "RF": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVM": LinearSVC(max_iter=2000, dual=False),
    "NB": GaussianNB(),
    "XGB": XGBClassifier(n_estimators=100, n_jobs=-1, eval_metric='logloss')
}

# 4. BENCHMARKING
results_f1 = []
model_names = []

print("Starting benchmark on stratified dataset...")
for name, model in models.items():
    print(f"Training {name}...")
    try:
        # Tree models don't strictly require scaling
        if name in ["DT", "RF", "XGB"]:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_std, y_train)
            preds = model.predict(X_test_std)
        
        score = f1_score(y_test, preds)
        results_f1.append(score)
        model_names.append(name)
    except Exception as e:
        print(f"Error training {name}: {e}")

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c']
bars = plt.bar(model_names, results_f1, color=colors, edgecolor='black', alpha=0.8)

# Add scores on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), 
             ha='center', va='bottom', fontweight='bold')

plt.title("Classifier Performance Comparison (F1-Score)", fontsize=14)
plt.ylabel("F1-Score", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('classifier_benchmark.png')
plt.show()

print("\n--- PERFORMANCE SUMMARY ---")
for n, s in zip(model_names, results_f1):
    print(f"{n}: {s:.4f}")