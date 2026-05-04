import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. SETUP
SEED = 42
np.random.seed(SEED)

def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')

features = list(set(df17.columns) & set(df18.columns))
features.remove('Label')

# 2. LOCAL ALIGNMENT
scaler17 = StandardScaler().fit(df17[features])
X17 = scaler17.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values

scaler18 = StandardScaler().fit(df18[features])

# 3. PRE-TRAIN EXPERT
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=SEED).fit(X17, y17)

# 4. EXPERIMENT LOOP WITH AVERAGING
percentages = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
mean_rf, mean_tl = [], []
ITERATIONS = 5 # We run each point 5 times to smooth the graph

test_df = df18.sample(int(0.2 * len(df18)), random_state=SEED)
X_test = scaler18.transform(test_df[features])
y_test = (test_df['Label'].str.lower() != 'benign').astype(int).values
test_expert_opin = expert_2017.predict_proba(X_test)[:, 1].reshape(-1, 1)

print("\nRunning Smoothed Comparison (5x Averaging)...")

for p in percentages:
    iter_rf, iter_tl = [], []
    sample_size = max(10, int((p / 100) * len(df18)))
    
    for i in range(ITERATIONS):
        train_df = df18.drop(test_df.index).sample(sample_size)
        X_train = scaler18.transform(train_df[features])
        y_train = (train_df['Label'].str.lower() != 'benign').astype(int).values
        
        # RF
        rf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
        iter_rf.append(f1_score(y_test, rf.predict(X_test)))
        
        # TL (Stacking)
        train_expert_opin = expert_2017.predict_proba(X_train)[:, 1].reshape(-1, 1)
        X_train_aug = np.hstack((X_train, train_expert_opin))
        X_test_aug = np.hstack((X_test, test_expert_opin))
        
        tl_stacker = xgb.XGBClassifier(n_estimators=50).fit(X_train_aug, y_train)
        iter_tl.append(f1_score(y_test, tl_stacker.predict(X_test_aug)))
        
    mean_rf.append(np.mean(iter_rf))
    mean_tl.append(np.mean(iter_tl))
    print(f"Data: {p:.3f}% | RF: {mean_rf[-1]:.3f} | TL: {mean_tl[-1]:.3f}")

# 5. PLOT
plt.figure(figsize=(10, 6))
plt.plot(percentages, mean_rf, label='Traditional RF (Baseline)', marker='o', color='#95a5a6', linestyle='--')
plt.plot(percentages, mean_tl, label='Transfer Learning (Expert Augmented)', marker='^', color='#2980b9', linewidth=2.5)
plt.title("Sample Efficiency Improvement through Transfer Learning", fontsize=14)
plt.xlabel("Percentage of 2018 Training Data (%)")
plt.ylabel("F1-Score (Averaged over 5 runs)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('smoothed_rising_curves.png')
plt.show()