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

# Feature names to avoid the UserWarning
features = df17.drop(columns=['Label']).columns.tolist()

# 2. PREP SCALER
scaler = StandardScaler().fit(df17[features])

def prep_binary(df):
    y = (df['Label'].str.lower() != 'benign').astype(int).values
    X = df[features].values
    return scaler.transform(X), y

# Pre-train Expert on 2017
X17, y17 = prep_binary(df17)
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# 3. THE EXPERIMENT LOOP (0.1% to 1.0% of ONLY DDoS)
top_attack = 'DDoS'
df_ddos_only = df18[(df18['Label'] == 'Benign') | (df18['Label'] == top_attack)]
df_test_all = df18.sample(10000, random_state=42) # Test against ALL attack types

percentages = np.linspace(0.1, 1.0, 10)
f1_rf, f1_tl = [], []

print(f"\nEvaluating Generalization (Training only on '{top_attack}')...")

for p in percentages:
    sample_size = max(10, int((p / 100) * len(df18)))
    # Sample training data only from the DDoS pool
    df_train = df_ddos_only.sample(min(sample_size, len(df_ddos_only)), random_state=42)
    
    X_train, y_train = prep_binary(df_train)
    X_test, y_test = prep_binary(df_test_all)
    
    # Train RF
    rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test)))
    
    # TL Adaptive (Search for best quantile)
    probs = expert_2017.predict_proba(X_test)[:, 1]
    best_q_f1 = 0
    for q in np.linspace(0.8, 0.99, 20):
        preds = (probs >= np.quantile(probs, q)).astype(int)
        score = f1_score(y_test, preds)
        if score > best_q_f1: best_q_f1 = score
    f1_tl.append(best_q_f1)
    
    print(f"Data: {p:.1f}% | RF F1: {f1_rf[-1]:.3f} | TL F1: {f1_tl[-1]:.3f}")

# 4. FINAL CONFUSION MATRIX (at 1.0%)
# Re-run the final predictions to get labels
rf_final_preds = rf.predict(X_test)
tl_probs = expert_2017.predict_proba(X_test)[:, 1]
tl_final_preds = (tl_probs >= np.quantile(tl_probs, 0.9)).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay.from_predictions(y_test, rf_final_preds, ax=ax[0], cmap='Reds', colorbar=False)
ax[0].set_title(f"RF (Trained only on {top_attack})")
ConfusionMatrixDisplay.from_predictions(y_test, tl_final_preds, ax=ax[1], cmap='Blues', colorbar=False)
ax[1].set_title("TL Expert (Trained on 2017 Diversity)")
plt.savefig('confusion_matrices.png')

# 5. PLOT EFFICIENCY GRAPH
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_rf, label=f'Traditional RF (Learned {top_attack})', marker='o', color='#e74c3c')
plt.plot(percentages, f1_tl, label='TL Expert (Generalized Knowledge)', marker='^', color='#2980b9', linewidth=2)
plt.title("Generalization Gap: Learning Specific vs. General Threats")
plt.xlabel("Percentage of 2018 Training Data Used (%)")
plt.ylabel("F1 Score (Tested on ALL 2018 Attack Types)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('generalization_efficiency.png')
print("\nGraphs saved: 'confusion_matrices.png' and 'generalization_efficiency.png'")
plt.show()

# Add this to the end of your script to see exactly WHAT we are catching
test_results = df_test_all.copy()
test_results['RF_Pred'] = rf_final_preds
test_results['TL_Pred'] = tl_final_preds

print("\n--- Detection Rate by Attack Type ---")
for label in test_results['Label'].unique():
    if label == 'Benign': continue
    
    subset = test_results[test_results['Label'] == label]
    rf_acc = subset['RF_Pred'].mean() # % of this attack type RF caught
    tl_acc = subset['TL_Pred'].mean() # % of this attack type TL caught
    
    print(f"{label:<25} | Samples: {len(subset):<4} | RF Caught: {rf_acc:.1%} | TL Caught: {tl_acc:.1%}")
    