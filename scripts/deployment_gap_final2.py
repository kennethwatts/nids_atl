import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. DATA LOADING
def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. THE EXPERT (2017)
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# 3. THE DEPLOYMENT GAP EXPERIMENT
# We test very small label counts where the "Cold Start" problem is real
labels = [0, 5, 10, 15, 20, 30, 50, 100]
res_rf, res_tl = [], []

test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'].str.lower() != 'benign').astype(int).values
probs_17 = expert_2017.predict_proba(X_test)[:, 1]

print("\nEvaluating the Deployment Gap...")
for n in labels:
    # Zero-Shot Baseline for TL
    f1_tl_0 = f1_score(y_test, (probs_17 >= np.quantile(probs_17, 0.9)).astype(int))
    
    if n == 0:
        res_rf.append(0)
        res_tl.append(f1_tl_0)
        print(f"Labels: {n:<3} | RF: 0.000 | TL: {res_tl[-1]:.3f} (Zero-Shot)")
        continue

    # Sample local labels
    train_df = df18.drop(test_df.index).sample(n, random_state=42)
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'].str.lower() != 'benign').astype(int).values
    
    # Check if we have both classes for RF to even function
    if len(np.unique(y_train)) < 2:
        res_rf.append(0)
    else:
        rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        res_rf.append(f1_score(y_test, rf.predict(X_test)))

    # TL: Weighted Opinion (Robust Transfer)
    # We combine 2017 knowledge with 2018 local observation
    if len(np.unique(y_train)) < 2:
        res_tl.append(f1_tl_0) # Fall back to Expert if no local attack seen
    else:
        local_18 = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        probs_18 = local_18.predict_proba(X_test)[:, 1]
        # Weighting: Gradually trust 2018 more, but keep 2017 as the "Anchor"
        combined = (0.6 * probs_17) + (0.4 * probs_18)
        res_tl.append(f1_score(y_test, (combined >= np.quantile(combined, 0.9)).astype(int)))
    
    print(f"Labels: {n:<3} | RF: {res_rf[-1]:.3f} | TL: {res_tl[-1]:.3f}")

# 4. PLOTTING
plt.figure(figsize=(10, 6))
plt.plot(labels, res_rf, label='Traditional ML (Requires Manual Labels)', marker='o', color='#e74c3c', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive Transfer Learning (Pre-trained)', marker='^', color='#2980b9', linewidth=2.5)

# Visualizing the Gap
plt.fill_between(labels, res_rf, res_tl, where=(np.array(res_tl) > np.array(res_rf)), 
                 color='green', alpha=0.2, label='Security Provided by TL')

plt.title("Bridging the 'Security Gap' in New Network Deployments", fontsize=14)
plt.xlabel("Number of Manual Labels Collected (Deployment Age)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('deployment_gap_final.png', dpi=300)
plt.show()