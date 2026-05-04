import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve

# 1. Architecture setup
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

def run_saturation_study(train_path, test_path, iterations=30):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train_full = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_train_full = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    sample_sizes = np.arange(100, 1100, 100)
    final_static = []
    final_best = []

    print("Starting Monte Carlo Simulation...")
    for size in sample_sizes:
        batch_static = []
        batch_best = []
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
                
                # Static 0.5 F1
                batch_static.append(f1_score(y_true, (probs > 0.5).astype(float), zero_division=0))
                
                # Best F1 (Analytical potential)
                p, r, _ = precision_recall_curve(y_true, probs)
                f1_s = (2 * p * r) / (p + r + 1e-8)
                batch_best.append(np.max(f1_s))
        
        final_static.append(np.mean(batch_static))
        final_best.append(np.mean(batch_best))

    # --- TIKZ OUTPUT LOGIC ---
    print("\n" + "="*30)
    print("TIKZ COORDINATES FOR LATEX")
    print("="*30)
    
    print("\n% Optimized Threshold (Blue Curve)")
    optimized_coords = "".join([f"({s},{m:.4f})" for s, m in zip(sample_sizes, final_best)])
    print(f"coordinates {{ {optimized_coords} }};")
    
    print("\n% Static Threshold (Red Curve)")
    static_coords = "".join([f"({s},{m:.4f})" for s, m in zip(sample_sizes, final_static)])
    print(f"coordinates {{ {static_coords} }};")
    print("="*30)

    # Standard Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, final_static, 'r--o', label='Static (0.5)')
    plt.plot(sample_sizes, final_best, 'b-s', label='Optimized (Analytical)')
    plt.title('Monte Carlo Saturation Study')
    plt.xlabel('Samples')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('saturation_analysis.png')
    plt.show()

if __name__ == "__main__":
    run_saturation_study("robust_2017_final.csv", "robust_2018_final.csv")