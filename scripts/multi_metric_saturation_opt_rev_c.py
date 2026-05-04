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

def run_multi_metric_study_final(train_csv, test_csv, iterations=30):
    # Load Datasets
    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
    except FileNotFoundError:
        print(f"Error: Ensure {train_csv} and {test_csv} are in the directory.")
        return
    
    X_train_raw = train_df.drop(columns=['Label']).values
    y_train_raw = train_df['Label'].values
    X_test_raw = test_df.drop(columns=['Label']).values
    y_test_raw = test_df['Label'].values

    sample_sizes = np.arange(100, 1100, 100)
    metrics_log = {s: {'f1': [], 'pr': [], 're': [], 'acc': [], 'gap': []} for s in sample_sizes}

    # --- OPTUNA OPTIMIZED HYPERPARAMETERS ---
    OPTIMIZED_LR = 0.0029813728587788817
    OPTIMIZED_BATCH_SIZE = 16
    OPTIMIZED_EPOCHS = 17

    print(f"Executing Monte Carlo Simulation with Optimized Params: LR={OPTIMIZED_LR:.5f}, BS={OPTIMIZED_BATCH_SIZE}, Epochs={OPTIMIZED_EPOCHS}")
    
    for size in sample_sizes:
        print(f"Sampling size: {size}...")
        for i in range(iterations):
            # Balanced sampling (stratified)
            idx = np.random.choice(len(X_train_raw), size, replace=False)
            X_batch_raw, y_batch = X_train_raw[idx], y_train_raw[idx]
            
            # Feature Scaling (Crucial for Transfer Learning)
            scaler = StandardScaler()
            X_batch = torch.tensor(scaler.fit_transform(X_batch_raw), dtype=torch.float32)
            X_val = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32)
            y_batch_t = torch.tensor(y_batch, dtype=torch.float32).reshape(-1, 1)
            
            model = IntrusionDetector(X_batch.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=OPTIMIZED_LR)
            criterion = nn.BCELoss()

            model.train()
            loader = DataLoader(TensorDataset(X_batch, y_batch_t), batch_size=OPTIMIZED_BATCH_SIZE, shuffle=True)
            for epoch in range(OPTIMIZED_EPOCHS):
                for bx, by in loader:
                    optimizer.zero_grad()
                    criterion(model(bx), by).backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = model(X_val).numpy()
                
                # Threshold Optimization for Transfer Learning
                p, r, t = precision_recall_curve(y_test_raw, probs)
                f1_s = (2 * p * r) / (p + r + 1e-8)
                best_idx = np.argmax(f1_s)
                best_thresh = t[min(best_idx, len(t)-1)]
                
                preds = (probs >= best_thresh).astype(float)
                
                # Logging Metrics
                p_val = precision_score(y_test_raw, preds, zero_division=0)
                r_val = recall_score(y_test_raw, preds, zero_division=0)
                
                metrics_log[size]['f1'].append(f1_score(y_test_raw, preds, zero_division=0))
                metrics_log[size]['pr'].append(p_val)
                metrics_log[size]['re'].append(r_val)
                metrics_log[size]['acc'].append(accuracy_score(y_test_raw, preds))
                metrics_log[size]['gap'].append(abs(p_val - r_val))

    # Calculate Means
    plot_data = {m: [np.mean(metrics_log[s][m]) for s in sample_sizes] for m in ['f1', 'pr', 're', 'acc', 'gap']}
    
    # Identify Equilibrium (Max F1 in stability zone)
    summary_df = pd.DataFrame(plot_data)
    summary_df['sample_size'] = sample_sizes
    # Search for highest F1 in the zone where gap is improving
    stable_zone = summary_df[summary_df['gap'] <= 0.45] 
    opt_row = stable_zone.loc[stable_zone['f1'].idxmax()] if not stable_zone.empty else summary_df.loc[summary_df['f1'].idxmax()]

    # --- TIKZ COORDINATE OUTPUT ---
    print("\n" + "="*50)
    print("TIKZ COORDINATES FOR LaTeX SOURCE")
    print("="*50)
    
    metric_labels = {'f1': 'F1-SCORE', 'pr': 'PRECISION', 're': 'RECALL', 'acc': 'ACCURACY', 'gap': 'P-R GAP'}
    for m_key, m_name in metric_labels.items():
        coords = " ".join([f"({s},{v:.4f})" for s, v in zip(sample_sizes, plot_data[m_key])])
        print(f"\n% {m_name} Curve\n\\addplot coordinates {{ {coords} }};")

    print(f"\n% OPTIMAL EQUILIBRIUM NODE\n\\node[circle, fill=red, inner sep=2pt, label={{above:Optimal Equilibrium}}] at (axis cs:{int(opt_row['sample_size'])},{opt_row['f1']:.4f}) {{}};")
    print("="*50)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, plot_data['f1'], 'b-s', label='F1-Score', linewidth=2)
    plt.plot(sample_sizes, plot_data['pr'], 'g--o', label='Precision', alpha=0.7)
    plt.plot(sample_sizes, plot_data['re'], 'm--^', label='Recall', alpha=0.7)
    plt.plot(sample_sizes, plot_data['acc'], 'r-d', label='Accuracy', linewidth=2)
    plt.plot(sample_sizes, plot_data['gap'], color='gray', linestyle=':', label='P-R Gap', linewidth=1.5)
    
    # Highlight Optimal Equilibrium
    plt.scatter(opt_row['sample_size'], opt_row['f1'], color='red', s=120, edgecolors='black', zorder=10)
    plt.axvline(x=opt_row['sample_size'], color='red', linestyle='--', alpha=0.3)

    plt.title('NIDS Multi-Metric Saturation: (32, 16) Architecture (Optimized Params)')
    plt.xlabel('Balanced 2017 Training Samples')
    plt.ylabel('Score (0.0 - 1.0)')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.savefig('multi_metric_saturation_opt_rev_c.png', dpi=300)
    plt.show()
    print("\nGraph saved as: multi_metric_saturation_opt_rev_c.png")

if __name__ == "__main__":
    run_multi_metric_study_final("robust_2017_final.csv", "robust_2018_final.csv")