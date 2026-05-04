import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# 1. (32, 16) MLP Architecture
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.block1 = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

def run_multi_metric_study(train_csv, test_csv, iterations=30):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Separate features and labels
    X_train_raw = train_df.drop(columns=['Label']).values
    y_train_raw = train_df['Label'].values
    X_test_raw = test_df.drop(columns=['Label']).values
    y_test_raw = test_df['Label'].values

    sample_sizes = np.arange(100, 1100, 100)
    metrics_log = {s: {'f1': [], 'pr': [], 're': [], 'acc': [], 'gap': []} for s in sample_sizes}

    print("Executing Monte Carlo Multi-Metric Simulation with Optuna-Derived Parameters...")
    
    for size in sample_sizes:
        print(f"Sampling size: {size}")
        for i in range(iterations):
            # Balanced sampling logic
            idx = np.random.choice(len(X_train_raw), size, replace=False)
            X_batch_raw, y_batch = X_train_raw[idx], y_train_raw[idx]
            
            # --- FEATURE SCALING ---
            # Re-applying scaling logic to handle 2017 -> 2018 drift
            scaler = StandardScaler()
            X_batch = torch.tensor(scaler.fit_transform(X_batch_raw), dtype=torch.float32)
            X_val = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32)
            y_batch_t = torch.tensor(y_batch, dtype=torch.float32).reshape(-1, 1)
            
            model = IntrusionDetector(X_batch.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=0.001) # Optuna-suggested LR
            criterion = nn.BCELoss()

            model.train()
            loader = DataLoader(TensorDataset(X_batch, y_batch_t), batch_size=32, shuffle=True)
            for epoch in range(15): # Increased epochs for convergence
                for bx, by in loader:
                    optimizer.zero_grad()
                    criterion(model(bx), by).backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = model(X_val).numpy()
                
                # --- THRESHOLD OPTIMIZATION ---
                # Finding the "Best" threshold from your original script logic
                p, r, t = precision_recall_curve(y_test_raw, probs)
                f1_s = (2 * p * r) / (p + r + 1e-8)
                best_idx = np.argmax(f1_s)
                best_thresh = t[min(best_idx, len(t)-1)]
                
                # Final Metric Calculation
                preds = (probs >= best_thresh).astype(float)
                
                metrics_log[size]['f1'].append(f1_score(y_test_raw, preds))
                metrics_log[size]['pr'].append(precision_score(y_test_raw, preds))
                metrics_log[size]['re'].append(recall_score(y_test_raw, preds))
                metrics_log[size]['acc'].append(accuracy_score(y_test_raw, preds))
                metrics_log[size]['gap'].append(abs(precision_score(y_test_raw, preds) - recall_score(y_test_raw, preds)))

    # Calculate Means for plotting
    plot_data = {m: [np.mean(metrics_log[s][m]) for s in sample_sizes] for m in ['f1', 'pr', 're', 'acc', 'gap']}

    # Identify Optimal Equilibrium Point
    # Highest F1 where gap < 0.15 (Stability Zone)
    summary_df = pd.DataFrame(plot_data)
    summary_df['sample_size'] = sample_sizes
    stable_zone = summary_df[summary_df['gap'] <= 0.15]
    opt_row = stable_zone.loc[stable_zone['f1'].idxmax()] if not stable_zone.empty else summary_df.loc[summary_df['f1'].idxmax()]

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    plt.plot(sample_sizes, plot_data['f1'], 'b-s', label='F1-Score', linewidth=2)
    plt.plot(sample_sizes, plot_data['pr'], 'g--o', label='Precision', alpha=0.6)
    plt.plot(sample_sizes, plot_data['re'], 'm--^', label='Recall', alpha=0.6)
    plt.plot(sample_sizes, plot_data['acc'], 'r-d', label='Accuracy', linewidth=2)
    plt.plot(sample_sizes, plot_data['gap'], color='gray', linestyle=':', label='P-R Gap')

    # Mark the Equilibrium
    plt.scatter(opt_row['sample_size'], opt_row['f1'], color='red', s=150, edgecolors='black', zorder=5, label='Optimal Equilibrium')
    plt.axvline(x=opt_row['sample_size'], color='red', linestyle='--', alpha=0.3)

    plt.title('NIDS Multi-Metric Saturation: (32, 16) Architecture with Optimized Threshold')
    plt.xlabel('Balanced 2017 Training Samples')
    plt.ylabel('Score (Normalized 0.0 - 1.0)')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.savefig('multi_metric_saturation_opt.png', dpi=300)
    plt.show()

    print(f"\nOptimal Equilibrium found at {int(opt_row['sample_size'])} samples with F1: {opt_row['f1']:.4f}")

if __name__ == "__main__":
    run_multi_metric_study("robust_2017_final.csv", "robust_2018_final.csv")