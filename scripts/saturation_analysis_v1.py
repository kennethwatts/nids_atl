import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve

# (32, 16) MLP Architecture
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

def run_saturation_study(train_csv, test_csv, iterations=30):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    X_train_full = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_train_full = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    sample_sizes = np.arange(100, 1100, 100)
    results_static = {size: [] for size in sample_sizes}
    results_best = {size: [] for size in sample_sizes}

    for size in sample_sizes:
        print(f"Running Monte Carlo for size: {size}")
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
                
                # Static 0.5 F1 (The one leveling at 0.175)
                f1_05 = f1_score(y_true, (probs > 0.5).astype(float), zero_division=0)
                results_static[size].append(f1_05)
                
                # 'Best' F1 (Finding the optimal threshold to see the 'true' saturation)
                precision, recall, thresholds = precision_recall_curve(y_true, probs)
                f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
                results_best[size].append(np.max(f1_scores))

    return sample_sizes, results_static, results_best

def plot_saturation(sizes, static, best):
    static_means = [np.mean(static[s]) for s in sizes]
    best_means = [np.mean(best[s]) for s in sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, static_means, 'r--o', label='Static Threshold (0.5)')
    plt.plot(sizes, best_means, 'b-s', label='Optimized Threshold (Target Potential)')
    plt.title('Monte Carlo Analysis: Pre-training Saturation (100-1000 Samples)')
    plt.xlabel('Balanced Training Samples')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('saturation_analysis.png')
    plt.show()

# Run the functional script
sizes, static, best = run_saturation_study("robust_2017_final.csv", "robust_2018_final.csv")
plot_saturation(sizes, static, best)