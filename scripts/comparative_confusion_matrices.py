import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD & CLEAN
def load_clean(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. PRE-TRAIN 2017 EXPERT
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 3. SETUP n=10 EXPERIMENT
n = 10
test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'] != 'benign').astype(int).values

# Balanced Sampling for 10 labels
pool_18 = df18.drop(test_df.index)
train_df = pd.concat([
    pool_18[pool_18['Label'] == 'benign'].sample(n//2, random_state=42),
    pool_18[pool_18['Label'] != 'benign'].sample(n//2, random_state=42)
])
X_train = scaler.transform(train_df[features])
y_train = (train_df['Label'] != 'benign').astype(int).values

# --- TRAIN THE THREE CASES ---

# Case A: Legacy (Zero-Shot) - Using Adaptive Thresholding
probs_17 = expert_2017.predict_proba(X_test)[:, 1]
y_pred_legacy = (probs_17 >= np.quantile(probs_17, 0.9)).astype(int)

# Case B: Local RF Only (n=10)
local_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train, y_train)
y_pred_local = local_rf.predict(X_test)

# Case C: Path Transfer TL (n=10)
X_train_leaves = expert_2017.apply(X_train)
X_test_leaves = expert_2017.apply(X_test)
tl_rf_student = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train_leaves, y_train)
y_pred_tl = tl_rf_student.predict(X_test_leaves)

# 4. VISUALIZE CONFUSION MATRICES
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cms = [
    (confusion_matrix(y_test, y_pred_legacy), "Legacy Expert (Zero-Shot)", "Oranges"),
    (confusion_matrix(y_test, y_pred_local), "Local RF (n=10 Labels)", "Reds"),
    (confusion_matrix(y_test, y_pred_tl), "Adaptive TL (n=10 Labels)", "Blues")
]

for i, (cm, title, cmap) in enumerate(cms):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Attack"])
    disp.plot(ax=axes[i], cmap=cmap, colorbar=False)
    axes[i].set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('comparative_confusion_matrices.png', dpi=300)
plt.show()

print("F1 Scores at n=10:")
print(f"Legacy: {f1_score(y_test, y_pred_legacy):.3f}")
print(f"Local:  {f1_score(y_test, y_pred_local):.3f}")
print(f"TL:     {f1_score(y_test, y_pred_tl):.3f}")