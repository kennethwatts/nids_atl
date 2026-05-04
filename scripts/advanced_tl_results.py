import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. SETUP & DATA
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
features = [f for f in df17.columns if f != 'Label']

# 2. THE TL ENGINE (PCA + XGB)
# We use 2017 to learn the "Structure of Malice"
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values

pca = PCA(n_components=10, random_state=SEED).fit(X17) # Learn feature relationships
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=SEED).fit(X17, y17)

# 3. EXPERIMENT
percentages = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
res_rf, res_tl = [], []

test_df = df18.sample(int(0.2 * len(df18)), random_state=SEED)
y_test = (test_df['Label'].str.lower() != 'benign').astype(int).values
X_test_raw = scaler.transform(test_df[features])
X_test_pca = pca.transform(X_test_raw) # Project 2018 into 2017's space
test_opin = expert_2017.predict_proba(X_test_raw)[:, 1].reshape(-1, 1)

print("\nRunning Advanced TL Comparison...")
for p in percentages:
    sample_size = max(10, int((p / 100) * len(df18)))
    train_df = df18.drop(test_df.index).sample(sample_size)
    y_train = (train_df['Label'].str.lower() != 'benign').astype(int).values
    X_train_raw = scaler.transform(train_df[features])
    
    # --- Traditional RF ---
    rf = RandomForestClassifier(n_estimators=50).fit(X_train_raw, y_train)
    res_rf.append(f1_score(y_test, rf.predict(X_test_raw)))
    
    # --- Advanced TL (PCA Projection + Expert Opinion) ---
    X_train_pca = pca.transform(X_train_raw)
    train_opin = expert_2017.predict_proba(X_train_raw)[:, 1].reshape(-1, 1)
    
    # Combine raw data, PCA features, and Expert Opinion
    X_train_tl = np.hstack((X_train_raw, X_train_pca, train_opin))
    X_test_tl = np.hstack((X_test_raw, X_test_pca, test_opin))
    
    tl_model = xgb.XGBClassifier(n_estimators=50).fit(X_train_tl, y_train)
    res_tl.append(f1_score(y_test, tl_model.predict(X_test_tl)))
    
    print(f"Data: {p:.3f}% | RF: {res_rf[-1]:.3f} | TL: {res_tl[-1]:.3f} | Gap: {res_tl[-1]-res_rf[-1]:.3f}")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(percentages, res_rf, label='Traditional ML (Random Forest)', marker='o', color='#95a5a6')
plt.plot(percentages, res_tl, label='Advanced Transfer Learning (PCA + Expert)', marker='^', color='#d35400', linewidth=2.5)
plt.title("Sample Efficiency: Advanced TL vs. Traditional ML", fontsize=14)
plt.xlabel("Percentage of 2018 Training Data (%)")
plt.ylabel("F1-Score")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('advanced_tl_results.png')
plt.show()