import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD MASTER DATASET
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)

# 2. DEFINE MAPPING LOGIC
# This groups specific attacks into their parent categories
category_map = {
    'DoS GoldenEye': 'DoS', 'DoS Hulk': 'DoS', 'DoS Slowhttptest': 'DoS', 'DoS slowloris': 'DoS',
    'DDoS': 'DDoS',
    'SSH-Bruteforce': 'Brute Force', 'FTP-Patator': 'Brute Force',
    'Web Attack - Brute Force': 'Web Attack', 'Web Attack - XSS': 'Web Attack', 'Web Attack - Sql Injection': 'Web Attack',
    'Bot': 'Botnet',
    'Infiltration': 'Infiltration'
}

df['Label'] = df['Label'].astype(str).str.strip()
df['Category'] = df['Label'].map(category_map).fillna('Benign')

# 3. CLEANING FEATURES
y = df['Category']
X = df.drop(columns=['Label', 'Category']).apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

# 4. ANALYSIS BY CATEGORY
categories = [c for c in y.unique() if c != 'Benign']
table_data = []

print(f"Aggregating signatures for {len(categories)} general categories...")

for cat in categories:
    mask = (y == 'Benign') | (y == cat)
    X_sub, y_sub = X[mask], (y[mask] == cat).astype(int)
    
    # Using 'balanced' ensures that even smaller categories like 'Infiltration' 
    # are weighed correctly against the 250k Benign samples.
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_sub, y_sub)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_3 = importances.head(3)
    
    table_data.append({
        'General Category': cat,
        'Primary Feature': f"{top_3.index[0]} ({top_3.values[0]:.3f})",
        'Secondary Feature': f"{top_3.index[1]} ({top_3.values[1]:.3f})",
        'Tertiary Feature': f"{top_3.index[2]} ({top_3.values[2]:.3f})"
    })

# 5. OUTPUT
final_table = pd.DataFrame(table_data).sort_values(by='General Category')
print("\n--- FINAL THESIS CATEGORY TABLE ---")
print(final_table.to_string(index=False))

# Optional: Export to LaTeX for your paper
print(final_table.to_latex(index=False))
