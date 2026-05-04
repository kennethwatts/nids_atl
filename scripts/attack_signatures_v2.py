import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD AND CLEAN
#file_path = "cic2018_training_av_lbl_100K_hdr.csv"
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path)

# STRIP SPACES AND UNIFY CASE (Fixes most "0.000" issues)
df['Label'] = df['Label'].astype(str).str.strip()
BENIGN_LABEL = 'Benign' # Ensure this matches your CSV exactly (check df['Label'].unique())

X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median()), axis=0) # Use median to avoid outlier skew
y = df['Label']

attacks = [a for a in y.unique() if a != BENIGN_LABEL]
feature_signatures = {}

print(f"Verified Labels: {y.unique()}")

for attack in attacks:
    mask = (y == BENIGN_LABEL) | (y == attack)
    X_sub = X[mask]
    y_sub = (y[mask] == attack).astype(int)
    
    # CHECK: Do we have enough samples of both?
    if y_sub.sum() < 5: 
        print(f"Skipping {attack}: Not enough attack samples (Found {y_sub.sum()})")
        continue
    
    # TRAIN WITH BALANCED WEIGHTS
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X_sub, y_sub)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_signatures[attack] = importances.sort_values(ascending=False).head(5)

# 2. GENERATE HEATMAP
if feature_signatures:
    sig_df = pd.DataFrame(feature_signatures).fillna(0)
    plt.figure(figsize=(14, 10))
    sns.heatmap(sig_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Attack-Feature Signatures (Importance Scores)")
    plt.tight_layout()
    plt.savefig('attack_signatures_v2.png')
    plt.show()
else:
    print("No signatures found. Check your BENIGN_LABEL variable.")