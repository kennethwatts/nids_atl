import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Loading your specific NIDS datasets
df_2017 = pd.read_csv('robust_2017_final.csv')
df_2018 = pd.read_csv('robust_2018_final.csv')

# Combine or select features/target based on your previous paper setup
# Assuming 'Label' is the target and features are already preprocessed
X = df_2017.drop(columns=['Label'])
y = df_2017['Label']

sample_sizes = np.arange(100, 1100, 100)
iterations = 50 
results = []

print("Starting Monte Carlo Simulation with Actual NIDS Data...")

for size in sample_sizes:
    print(f"Processing sample size: {size}...")
    for i in range(iterations):
        # Create balanced subsample for this iteration
        # (Assuming binary classification: 0 = Normal, 1 = Attack)
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=size, stratify=y)
        
        # Define the (32, 16) architecture you've selected as optimal
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=i)
        clf.fit(X_sub, y_sub)
        
        # Test against the 2018 robust dataset for transfer learning evaluation
        y_pred = clf.predict(df_2018.drop(columns=['Label']))
        y_true = df_2018['Label']
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        accuracy = accuracy_score(y_true, y_pred)
        pr_gap = abs(precision - recall)
        
        results.append({
            'sample_size': size,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'pr_gap': pr_gap
        })

# --- Processing & Equilibrium Identification ---
df_results = pd.DataFrame(results)
summary = df_results.groupby('sample_size').mean().reset_index()

# Find the Optimal Equilibrium (where P-R Gap stabilizes < 0.02)
threshold = 0.02
stable_points = summary[summary['pr_gap'] <= threshold]
opt_point = stable_points.iloc[0] if not stable_points.empty else summary.iloc[summary['pr_gap'].idxmin()]

eq_size = int(opt_point['sample_size'])
eq_f1 = opt_point['f1']

# --- Output TikZ for LaTeX ---
print("\n--- TikZ Coordinates for LaTeX ---")
for metric in ['f1', 'precision', 'recall', 'accuracy', 'pr_gap']:
    coords = " ".join([f"({int(row['sample_size'])}, {row[metric]:.4f})" for _, row in summary.iterrows()])
    print(f"% {metric.upper()}:\n\\addplot coordinates {{ {coords} }};")

# --- Plotting ---
plt.figure(figsize=(10, 6))
metrics_to_plot = {'f1': 'F1-Score', 'precision': 'Precision', 'recall': 'Recall', 'accuracy': 'Accuracy'}

for key, label in metrics_to_plot.items():
    plt.plot(summary['sample_size'], summary[key], label=label, marker='o', alpha=0.8)

# Add P-R Gap as a reference for stability
plt.plot(summary['sample_size'], summary['pr_gap'], label='P-R Gap (Stability)', linestyle='--', color='gray')

# Annotate Optimal Equilibrium
plt.axvline(x=eq_size, color='red', linestyle=':', alpha=0.5)
plt.scatter(eq_size, eq_f1, color='red', edgecolors='black', s=80, zorder=5, label='Optimal Equilibrium')

plt.title('NIDS Multi-Metric Saturation (32, 16 Architecture)')
plt.xlabel('Balanced Training Samples')
plt.ylabel('Performance Metric Score')
plt.legend(loc='lower right')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Save with the new filename
plt.savefig('multi_metric_saturation_opt.png', dpi=300)
print(f"\nGraph saved as: multi_metric_saturation_opt.png")
plt.show()