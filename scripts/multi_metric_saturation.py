import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score

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
    
    X_train_full = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_train_full = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    sample_sizes = np.arange(100, 1100, 100)
    metrics_log = {s: {'f1': [], 'pr': [], 're': [], 'acc': []} for s in sample_sizes}

    print("Executing Monte Carlo Multi-Metric Simulation...")
    for size in sample_sizes:
        print(f"Sampling size: {size}")
        for i in range(iterations):
            idx = np.random.choice(len(X_train_full), size, replace=False)
            X_batch, y_batch = X_train_full[idx], y_train_full[idx]
            
            model = IntrusionDetector(X_train_full.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            model.train()
            loader = DataLoader(TensorDataset(X_batch, y_batch), batch_size=32, shuffle=True)
            for epoch in range(10):
                for bx, by in loader:
                    optimizer.zero_grad()
                    criterion(model(bx), by).backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = model(X_test).numpy()
                y_true = y_test.numpy()
                
                # Find the optimal threshold
                p, r, t = precision_recall_curve(y_true, probs)
                f1_s = (2 * p * r) / (p + r + 1e-8)
                best_idx = np.argmax(f1_s)
                
                metrics_log[size]['f1'].append(f1_s[best_idx])
                metrics_log[size]['pr'].append(p[best_idx])
                metrics_log[size]['re'].append(r[best_idx])
                
                best_thresh = t[min(best_idx, len(t)-1)]
                preds = (probs >= best_thresh).astype(float)
                metrics_log[size]['acc'].append(accuracy_score(y_true, preds))

    # Calculate Means
    plot_data = {m: [np.mean(metrics_log[s][m]) for s in sample_sizes] for m in ['f1', 'pr', 're', 'acc']}

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, plot_data['f1'], 'b-s', label='F1-Score', linewidth=2)
    plt.plot(sample_sizes, plot_data['pr'], 'g--o', label='Precision', alpha=0.7)
    plt.plot(sample_sizes, plot_data['re'], 'm--^', label='Recall', alpha=0.7)
    plt.plot(sample_sizes, plot_data['acc'], 'r-d', label='Accuracy', linewidth=2)

    plt.title('Monte Carlo: Optimized Threshold Metrics (100-1000 Samples)')
    plt.xlabel('Balanced 2017 Training Samples')
    plt.ylabel('Score (Normalized 0.0 - 1.0)')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.savefig('multi_metric_saturation.png', dpi=300)
    plt.show()

    # Print TikZ Coordinates for LaTeX
    print("\n" + "="*40)
    print("TIKZ COORDINATES FOR MULTI-METRIC GRAPH")
    print("="*40)
    for metric in ['f1', 'pr', 're', 'acc']:
        coords = "".join([f"({s},{v:.4f})" for s, v in zip(sample_sizes, plot_data[metric])])
        print(f"\n% {metric.upper()} Curve\ncoordinates {{ {coords} }};")

if __name__ == "__main__":
    run_multi_metric_study("robust_2017_final.csv", "robust_2018_final.csv")