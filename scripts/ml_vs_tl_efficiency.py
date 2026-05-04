import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# 1. SETUP & SEEDING
SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Traditional ML vs Transfer Learning Comparison')
parser.add_argument('file_2017', help='Path to 2017 dataset (Pre-train)')
parser.add_argument('file_2018', help='Path to 2018 dataset (Train/Test)')
args = parser.parse_args()

# 2. DATA LOADING
def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df.drop(columns=['Label']).values
    y = df['Label'].astype(int).values
    return X, y

X17, y17 = load_data(args.file_2017)
X18, y18 = load_data(args.file_2018)

# Standardize based on 2017 (for TL) and 2018 (for Traditional ML)
scaler_tl = StandardScaler().fit(X17)
X17_scaled = scaler_tl.transform(X17)
X18_scaled_tl = scaler_tl.transform(X18)

scaler_trad = StandardScaler().fit(X18)
X18_scaled_trad = scaler_trad.transform(X18)

# 3. PRE-TRAINING THE TL MODEL (XGBoost base)
print("Pre-training TL model on 2017 data...")
tl_base_model = xgb.XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='logloss')
tl_base_model.fit(X17_scaled, y17)
# Save the model to a temporary file to simulate "weight transfer"
tl_base_model.save_model("base_2017.json")

# 4. EXPERIMENT LOOP (1% to 10%)
percentages = np.arange(1, 11)
f1_rf = []
f1_xgb = []
f1_tl = []

# Reserve 20% of 2018 for a consistent final test set
test_size = int(0.2 * len(X18))
indices = np.arange(len(X18))
np.random.shuffle(indices)
test_idx = indices[:test_size]
train_pool_idx = indices[test_size:]

X_test_trad = X18_scaled_trad[test_idx]
X_test_tl = X18_scaled_tl[test_idx]
y_test = y18[test_idx]

for p in percentages:
    # Calculate sample size
    sample_size = int((p / 100) * len(X18))
    train_idx = np.random.choice(train_pool_idx, sample_size, replace=False)
    
    X_train_trad = X18_scaled_trad[train_idx]
    X_train_tl = X18_scaled_tl[train_idx]
    y_train = y18[train_idx]
    
    # --- Traditional ML: Random Forest ---
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_trad, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test_trad)))
    
    # --- Traditional ML: XGBoost ---
    trad_xgb = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    trad_xgb.fit(X_train_trad, y_train)
    f1_xgb.append(f1_score(y_test, trad_xgb.predict(X_test_trad)))
    
    # --- Transfer Learning: XGBoost (Update) ---
    # We load the 2017 weights and "fine-tune" on the 1-10% 2018 data
    tl_model = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    tl_model.load_model("base_2017.json")
    # Using 'process_type': 'update' allows XGBoost to adjust existing trees to new data
    tl_model.fit(X_train_tl, y_train, xgb_model="base_2017.json")
    f1_tl.append(f1_score(y_test, tl_model.predict(X_test_tl)))
    
    print(f"Progress: {p}% | RF: {f1_rf[-1]:.3f} | XGB: {f1_xgb[-1]:.3f} | TL: {f1_tl[-1]:.3f}")

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_rf, label='Traditional ML (Random Forest)', marker='o', color='#7f8c8d')
plt.plot(percentages, f1_xgb, label='Traditional ML (XGBoost)', marker='s', color='#2ecc71')
plt.plot(percentages, f1_tl, label='Transfer Learning (2017 -> 2018)', marker='^', color='#3498db', linewidth=2)

plt.title("Sample Efficiency: Traditional ML vs. Transfer Learning", fontsize=14)
plt.xlabel("Percentage of 2018 Data Used for Training (%)", fontsize=12)
plt.ylabel("F1-Score on 2018 Test Set", fontsize=12)
plt.xticks(percentages)
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()

plt.savefig('ml_vs_tl_efficiency.png', dpi=300)
print("\nGraph saved as 'ml_vs_tl_efficiency.png'")
plt.show()