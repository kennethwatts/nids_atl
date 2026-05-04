import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD & CLEAN
def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. PRE-TRAIN 2017 (THE ARCHITECT)
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 3. EXPERIMENT
labels = [0, 10, 20, 50, 100]
res_local_rf, res_legacy, res_tl_rf = [], [], []

test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'] != 'benign').astype(int).values

print("\nRunning RF-Based Path Transfer Comparison...")
for n in labels:
    # 1. Legacy Baseline (Adaptive)
    p17 = expert_2017.predict_proba(X_test)[:, 1]
    res_legacy.append(f1_score(y_test, (p17 >= np.quantile(p17, 0.9)).astype(int)))

    if n == 0:
        res_local_rf.append(0)
        res_tl_rf.append(res_legacy[-1])
        continue

    # Balanced Sampling
    pool_18 = df18.drop(test_df.index)
    train_df = pd.concat([
        pool_18[pool_18['Label'] == 'benign'].sample(n//2, random_state=42),
        pool_18[pool_18['Label'] != 'benign'].sample(n//2, random_state=42)
    ])
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'] != 'benign').astype(int).values
    
    # 2. Local RF Only
    local_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train, y_train)
    res_local_rf.append(f1_score(y_test, local_rf.predict(X_test)))

    # 3. TL: Path Transfer using RF as the student
    X_train_leaves = expert_2017.apply(X_train)
    X_test_leaves = expert_2017.apply(X_test)
    
    tl_rf_student = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train_leaves, y_train)
    res_tl_rf.append(f1_score(y_test, tl_rf_student.predict(X_test_leaves)))
    
    print(f"Labels: {n:<3} | Legacy: {res_legacy[-1]:.3f} | Local RF: {res_local_rf[-1]:.3f} | TL RF: {res_tl_rf[-1]:.3f}")

# 4. CALCULATE GAIN AREA
x_vals = np.array(labels)
y_tl = np.array(res_tl_rf)
y_local = np.array(res_local_rf)
# Use trapezoid for area calculation
gain_area = np.trapezoid(y_tl - y_local, x_vals)

# 5. PLOT
plt.figure(figsize=(10, 6))
plt.plot(labels, res_legacy, label='Legacy (2017 Expert)', color='#f39c12', linestyle=':', linewidth=2)
plt.plot(labels, res_local_rf, label='Local RF (2018 Only)', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels, res_tl_rf, label='Adaptive TL (RF Path Transfer)', color='#2980b9', marker='^', linewidth=3)

plt.fill_between(labels, res_local_rf, res_tl_rf, where=(y_tl >= y_local), 
                 color='green', alpha=0.2, interpolate=True, 
                 label=f'Reduced Vulnerability Window (Area: {gain_area:.2f})')

plt.title("RF-Based Sample Efficiency: Path Transfer Gain", fontsize=14, fontweight='bold')
plt.xlabel("Number of Manual Labels", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rf_path_transfer_gain.png', dpi=300)
plt.show()

print(f"Total RF Transfer Gain Area: {gain_area:.4f}")