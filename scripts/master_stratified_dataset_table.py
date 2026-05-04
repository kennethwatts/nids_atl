import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. ROBUST LOADING
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)

# 2. CLEANING
df['Label'] = df['Label'].astype(str).str.strip()
y = df['Label']
X = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

# 3. ANALYSIS
attacks = [a for a in y.unique() if a.lower() != 'benign']
table_data = []

print(f"Generating signatures for {len(attacks)} attack types...")

for attack in attacks:
    mask = (y.str.lower() == 'benign') | (y == attack)
    X_sub, y_sub = X[mask], (y[mask] == attack).astype(int)
    
    if y_sub.sum() < 5: continue

    # Use 'balanced' to ensure rare samples carry weight
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_sub, y_sub)
    
    # Extract Top 3 Features
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_3 = importances.head(3)
    
    table_data.append({
        'Attack Type': attack,
        'Primary Feature': f"{top_3.index[0]} ({top_3.values[0]:.3f})",
        'Secondary Feature': f"{top_3.index[1]} ({top_3.values[1]:.3f})",
        'Tertiary Feature': f"{top_3.index[2]} ({top_3.values[2]:.3f})"
    })

# 4. OUTPUT TABLE
final_table = pd.DataFrame(table_data)
print("\n--- FINAL THESIS FEATURE TABLE ---")
print(final_table.to_string(index=False))

# Optional: Export to LaTeX for your paper
print(final_table.to_latex(index=False))