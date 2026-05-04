import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD DATA
# Note: Use the raw labels version of the file to see specific attack names
#file_path = "cic2018_training_av_lbl_100K_hdr.csv"
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path)

# Sanitize
X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
y = df['Label']

# 2. IDENTIFY ATTACK TYPES
attacks = [a for a in y.unique() if str(a).upper() != 'BENIGN']
feature_signatures = {}

print(f"Analyzing signatures for {len(attacks)} attack types...")

for attack in attacks:
    # Create a binary target: This specific attack vs Benign
    mask = (y == 'Benign') | (y == attack)
    X_sub = X[mask]
    y_sub = (y[mask] == attack).astype(int)
    
    # Train a quick RF to find specific features for this attack
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_sub, y_sub)
    
    # Get top 5 features
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_signatures[attack] = importances.sort_values(ascending=False).head(5)

# 3. VISUALIZATION (Heatmap of Signatures)
sig_df = pd.DataFrame(feature_signatures).fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(sig_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Importance Score'})
plt.title("Attack-Feature Signature Map: Representative Features by Attack Type", fontsize=15)
plt.xlabel("Attack Type", fontsize=12)
plt.ylabel("Network Feature", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('attack_signatures.png')
plt.show()

# 4. PRINT SUMMARY FOR DEFENSE
for attack, features in feature_signatures.items():
    print(f"\n--- Top Representative Features for {attack} ---")
    for feat, score in features.items():
        print(f"- {feat}: {score:.4f}")