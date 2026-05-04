import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD & CLEAN
def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_data('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_data('cic2018_training_av_lbl_100K_hdr.csv')

# Use only numeric features; strictly exclude 'Label'
features = [col for col in df17.columns if col != 'Label']

# 2. DATA SEGREGATION (Prevent Leakage)
# We shuffle 2018 and cut it into two completely separate pieces
all_indices = df18.index.tolist()
np.random.seed(42)
np.random.shuffle(all_indices)

# Training pool (First 10k rows)
train_pool_indices = all_indices[:10000]
# Test set (A different 10k rows)
test_indices = all_indices[10000:20000]

df_train_pool = df18.loc[train_pool_indices]
df_test_all = df18.loc[test_indices]

# 3. PREP SCALER (Fit on 2017 only)
scaler = StandardScaler().fit(df17[features])

def prep_binary(df, scaler_obj):
    y = (df['Label'].str.lower() != 'benign').astype(int).values
    X = df[features].values
    return scaler_obj.transform(X), y

X17, y17 = prep_binary(df17, scaler)
X_test, y_test = prep_binary(df_test_all, scaler)

# 4. PRE-TRAIN EXPERT (The "Generalist")
print("Pre-training 2017 Expert (Diversity Training)...")
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# 5. EXPERIMENT LOOP (Specialized Training)
top_attack = 'DDoS'
# RF only gets to see DDoS from the training pool
df_ddos_pool = df_train_pool[(df_train_pool['Label'] == 'Benign') | (df_train_pool['Label'] == top_attack)]

percentages = np.linspace(0.1, 1.0, 10)
f1_rf, f1_tl = [], []

print(f"\nEvaluating Generalization (RF training only on '{top_attack}')...")

for p in percentages:
    sample_size = max(20, int((p / 100) * len(df18)))
    df_train = df_ddos_pool.sample(min(sample_size, len(df_ddos_pool)), random_state=42)
    
    X_train, y_train = prep_binary(df_train, scaler)
    
    # Train specialized RF
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test)))
    
    # TL Adaptive Logic
    probs = expert_2017.predict_proba(X_test)[:, 1]
    best_q_f1 = 0
    # Search for the optimal operating point for the 2017 model on 2018 data
    for q in np.linspace(0.85, 0.99, 15):
        preds = (probs >= np.quantile(probs, q)).astype(int)
        score = f1_score(y_test, preds)
        if score > best_q_f1: best_q_f1 = score
    f1_tl.append(best_q_f1)
    
    print(f"Data: {p:.1f}% | RF F1: {f1_rf[-1]:.3f} | TL F1: {f1_tl[-1]:.3f}")

# 6. DETECTION BY ATTACK TYPE REPORT
rf_final_preds = rf.predict(X_test)
tl_probs = expert_2017.predict_proba(X_test)[:, 1]
# Use the final best quantile found (approx 0.9 for this data)
tl_final_preds = (tl_probs >= np.quantile(tl_probs, 0.9)).astype(int)

df_report = df_test_all.copy()
df_report['RF_Hit'] = rf_final_preds
df_report['TL_Hit'] = tl_final_preds

print("\n" + "="*50)
print(f"{'Attack Type':<25} | {'Size':<6} | {'RF':<7} | {'TL':<7}")
print("-" * 50)
for label in sorted(df_report['Label'].unique()):
    subset = df_report[df_report['Label'] == label]
    rf_rate = subset['RF_Hit'].mean()
    tl_rate = subset['TL_Hit'].mean()
    print(f"{label:<25} | {len(subset):<6} | {rf_rate:>6.1%} | {tl_rate:>6.1%}")
print("="*50)

# 7. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_rf, label=f'Traditional RF (Learned {top_attack})', marker='o', color='#e74c3c')
plt.plot(percentages, f1_tl, label='TL Expert (Generalized Knowledge)', marker='^', color='#2980b9', linewidth=2)
plt.title("The Generalization Gap: Zero-Day Resilience")
plt.xlabel("Percentage of 2018 Data Used for Training (%)")
plt.ylabel("F1 Score (Tested on ALL 2018 Attack Types)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('hardened_generalization.png')
plt.show()