import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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

# 2. Functional Monte Carlo Scaling
def run_pretraining_saturation(train_path, test_path, iterations=30):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train_full = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_train_full = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    # Scaling from 100 to 1000 in 100-sample increments
    sample_sizes = np.arange(100, 1100, 100)
    means = []
    stds = []

    for size in sample_sizes:
        batch_f1s = []
        print(f"Sampling {size} records...")
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
                preds = (model(X_test) > 0.5).float()
                batch_f1s.append(f1_score(y_test.numpy(), preds.numpy(), zero_division=0))
        
        means.append(np.mean(batch_f1s))
        stds.append(np.std(batch_f1s))

    # 3. Plotting Logic
    plt.figure(figsize=(10, 6))
    plt.errorbar(sample_sizes, means, yerr=stds, fmt='-o', color='darkblue', 
                 ecolor='lightblue', elinewidth=3, capsize=0, label='Novice ML Performance')
    
    plt.axhline(y=max(means), color='red', linestyle='--', alpha=0.5, label=f'Saturation Floor (~{max(means):.3f})')
    
    plt.title('Pre-training Saturation: Source Data vs. Target F1 Score')
    plt.xlabel('Number of Balanced 2017 Training Samples')
    plt.ylabel('F1 Score on 2018 Target Domain')
    plt.ylim(0, 0.25) # Adjusted for the 100-1000 sample leveling observed
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig('pretraining_saturation_plot.png', dpi=300)
    plt.show()

    return sample_sizes, means

if __name__ == "__main__":
    run_pretraining_saturation("robust_2017_final.csv", "robust_2018_final.csv")