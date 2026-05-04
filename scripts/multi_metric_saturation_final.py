import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- (32, 16) Architecture as defined in the Adaptive TL framework ---
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

def run_final_saturation_study(train_csv, test_csv, iterations=30):
    # --- OPTUNA-OPTIMIZED PARAMETERS ---
    BEST_LR = 0.0013829750664045026
    BEST_BS = 64
    BEST_EPOCHS = 20
    BEST_Q = 0.9010240747074492  # Tuned q-Quantile for AQT sensitivity [cite: 166]

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    X_train_raw = train_df.drop(columns=['Label']).values
    y_train_raw = train_df['Label'].values
    X_test_raw = test_df.drop(columns=['Label']).values
    y_test_raw = test_df['Label'].values

    sample_sizes = np.arange(100, 1100, 100)
    metrics_log = {s: {'f1_opt': [], 'f1_static': [], 'pr': [], 're': [], 'acc': [], 'gap': []} for s in sample_sizes}

    print(f"Generating Final Results using Tuned q-Quantile: {BEST_Q:.4f}")
    
    for size in sample_sizes:
        print(f"Processing size: {size}...")
        for i in range(iterations):
            # Monte Carlo sampling to account for stochastic selection [cite: 205, 273]
            idx = np.random.choice(len(X_train_raw), size, replace=False)
            X_batch_raw, y_batch = X_train_raw[idx], y_train_raw[idx]
            
            scaler = StandardScaler()
            X_batch = torch.tensor(scaler.fit_transform(X_batch_raw), dtype=torch.float32)
            X_val = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32)
            y_batch_t = torch.tensor(y_batch, dtype=torch.float32).reshape(-1, 1)
            
            model = IntrusionDetector(X_batch.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
            criterion = nn.BCELoss()

            model.train()
            loader = DataLoader(TensorDataset(X_batch, y_batch_t), batch_size=BEST_BS, shuffle=True)
            for epoch in range(BEST_EPOCHS):
                for bx, by in loader:
                    optimizer.zero_grad()
                    criterion(model(bx), by).backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = model(X_val).numpy().flatten()
                
                # --- APPLYING TUNED AQT THRESHOLD [cite: 151, 383] ---
                thresh_opt = np.quantile(probs, BEST_Q)
                preds_opt = (probs >= thresh_opt).astype(float)
                
                # --- STATIC THRESHOLD (0.5) FOR COMPARISON [cite: 167, 385] ---
                preds_static = (probs >= 0.5).astype(float)
                
                # Optimized Metrics
                p_val = precision_score(y_test_raw, preds_opt, zero_division=0)
                r_val = recall_score(y_test_raw, preds_opt, zero_division=0)
                
                metrics_log[size]['f1_opt'].append(f1_score(y_test_raw, preds_opt, zero_division=0))
                metrics_log[size]['f1_static'].append(f1_score(y_test_raw, preds_static, zero_division=0))
                metrics_log[size]['pr'].append(p_val)
                metrics_log[size]['re'].append(r_val)
                metrics_log[size]['acc'].append(accuracy_score(y_test_raw, preds_opt))
                metrics_log[size]['gap'].append(abs(p_val - r_val))

    # Calculate Means for plotting and TikZ coordinates [cite: 37, 410]
    plot_data = {m: [np.mean(metrics_log[s][m]) for s in sample_sizes] for m in ['f1_opt', 'f1_static', 'pr', 're', 'acc', 'gap']}

    # --- TIKZ COORDINATES FOR LaTeX  ---
    print("\n" + "="*50)
    print("TIKZ COORDINATES FOR ACADEMIC PLOTS")
    print("="*50)
    for m in ['f1_opt', 'f1_static', 'pr', 're', 'gap']:
        coords = " ".join([f"({s},{v:.4f})" for s,v in zip(sample_sizes, plot_data[m])])
        print(f"\n% {m.upper()} Curve\n\\addplot coordinates {{ {coords} }};")

    # Plotting
    plt.figure(figsize=(10,7))
    plt.plot(sample_sizes, plot_data['f1_opt'], 'b-s', label='F1 (Optimized Threshold)', linewidth=2)
    plt.plot(sample_sizes, plot_data['pr'], 'g--o', label='Precision', alpha=0.7)
    plt.plot(sample_sizes, plot_data['re'], 'm--^', label='Recall', alpha=0.7)
    plt.plot(sample_sizes, plot_data['gap'], color='gray', linestyle=':', label='P-R Gap', alpha=0.8)
    plt.plot(sample_sizes, plot_data['f1_static'], 'r-.d', label='F1 (Static 0.5)', alpha=0.6)

    plt.title('Final Multi-Metric Saturation Analysis (AQT Optimized)')
    plt.xlabel('Balanced 2017 Training Samples')
    plt.ylabel('Score (0.0 - 1.0)')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- MODIFIED LEGEND LOCATION ---
    plt.legend(loc='upper left', frameon=True, shadow=True)
    
    plt.savefig('multi_metric_saturation_final.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_final_saturation_study("robust_2017_final.csv", "robust_2018_final.csv")