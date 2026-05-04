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

def run_optimized_metrics_study(train_csv, test_csv, iterations=30):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    X_train_full = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_train_full = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    sample_sizes = np.arange(100, 1100, 100)
    
    # Storage for averages
    metrics = {s: {'f1': [], 'pr': [], 're': [], 'acc': []} for s in sample_sizes}

    print("Starting Optimized Metrics Monte Carlo...")
    for size in sample_sizes:
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
                
                # Calculate PR Curve to find the optimal operating point (Threshold)
                precision, recall, thresholds = precision_recall_curve(y_true, probs)
                f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
                
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
                
                # Record metrics at that specific best threshold
                metrics[size]['f1'].append(f1_scores[best_idx])
                metrics[size]['pr'].append(precision[best_idx])
                metrics[size]['re'].append(recall[best_idx])
                
                # Accuracy at the best threshold
                preds = (probs >= best_threshold).astype(float)
                metrics[size]['acc'].append(accuracy_score(y_true, preds))

    # Output Data Table
    print("\n" + "="*80)
    print(f"{'Size':<6} | {'F1 Mean':<10} | {'Prec Mean':<10} | {'Rec Mean':<10} | {'Acc Mean':<10}")
    print("-" * 80)
    
    plot_f1 = []
    for s in sample_sizes:
        m_f1 = np.mean(metrics[s]['f1'])
        m_pr = np.mean(metrics[s]['pr'])
        m_re = np.mean(metrics[s]['re'])
        m_acc = np.mean(metrics[s]['acc'])
        plot_f1.append(m_f1)
        print(f"{s:<6} | {m_f1:<10.4f} | {m_pr:<10.4f} | {m_re:<10.4f} | {m_acc:<10.4f}")
    
    # Plotting only the Optimized Curve
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, plot_f1, 'b-s', linewidth=2, label='Optimized Threshold Potential')
    plt.title('Monte Carlo: Optimized Threshold Scaling (100-1000 Samples)')
    plt.xlabel('Balanced 2017 Training Samples')
    plt.ylabel('Mean F1-Score on 2018 Target')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('optimized_saturation_only.png')
    plt.show()

if __name__ == "__main__":
    run_optimized_metrics_study("robust_2017_final.csv", "robust_2018_final.csv")