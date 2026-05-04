import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD WITH ENCODING FIX
file_path = "master_stratified_dataset.csv"

# 'utf-8-sig' handles the \ufeff error automatically
# low_memory=False stops the DtypeWarning
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)

# 2. AGGRESSIVE CLEANING
# Clean labels first
df['Label'] = df['Label'].astype(str).str.strip()
y = df['Label']

# Separate features and force everything to numeric
X = df.drop(columns=['Label'])

# Convert all columns to numeric, turning strings/errors into NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Drop columns that are entirely NaN (like Timestamp or ID columns that became strings)
X = X.dropna(axis=1, how='all')

# Now handle remaining NaNs and Infinites
X = X.replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

print(f"Data Cleaned. Feature count: {X.shape[1]}")
print(f"Attack types found: {y.unique()}")

# 3. RUN SIGNATURE ANALYSIS (Same as before)
attacks = [a for a in y.unique() if a.lower() != 'benign']
feature_signatures = {}

for attack in attacks:
    mask = (y.str.lower() == 'benign') | (y == attack)
    X_sub = X[mask]
    y_sub = (y[mask] == attack).astype(int)
    
    if y_sub.sum() < 10: continue

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_sub, y_sub)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_signatures[attack] = importances.sort_values(ascending=False).head(5)

# 4. PLOT
if feature_signatures:
    plt.figure(figsize=(14, 10))
    sns.heatmap(pd.DataFrame(feature_signatures).fillna(0), annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Master Attack Signatures")
    plt.savefig('master_signatures_fixed.png')
    plt.show()