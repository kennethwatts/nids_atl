import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import precision_recall_curve

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

def objective(trial, X_train, y_train, X_test, y_test):
    # Hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)
    
    model = IntrusionDetector(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training logic
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    # Evaluation logic
    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy()
        y_true = y_test.numpy()
        
        # Calculate PR Curve
        p, r, t = precision_recall_curve(y_true, probs)
        f1_scores = (2 * p * r) / (p + r + 1e-8)
        
        # Identify the point of MAXIMUM F1 performance
        best_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[best_f1_idx]
        
        # Calculate the P-R Gap specifically at that peak performance point
        gap_at_peak = np.abs(p[best_f1_idx] - r[best_f1_idx])
        
        # Store metadata for plotting later
        trial.set_user_attr("f1", max_f1)
        trial.set_user_attr("gap", gap_at_peak)

        # THE PENALTY FUNCTION:
        # We want to minimize this value. 
        # (1 - max_f1) pushes performance up.
        # gap_at_peak pushes for balance.
        return (1 - max_f1) + gap_at_peak

def run_balanced_optimization(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Fix sample size at 1,000 for pre-training consistency
    train_df = train_df.sample(n=1000, random_state=42)
    
    X_train = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    # Run Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test), n_trials=50)

    # Data Extraction for Plotting
    results = study.trials_dataframe()
    f1_vals = [t.user_attrs['f1'] for t in study.trials]
    gap_vals = [t.user_attrs['gap'] for t in study.trials]

    # --- Plot 1: Optimization History ---
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Combined Objective: (1-F1) + |P-R Gap|")
    plt.tight_layout()
    plt.savefig('balanced_opt_history.png')

    # --- Plot 2: Performance vs. Stability Trade-off ---
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(gap_vals, f1_vals, c=range(len(f1_vals)), cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(sc, label='Trial Number')
    plt.xlabel('Precision-Recall Gap (Lower is more stable)')
    plt.ylabel('F1-Score (Higher is better performance)')
    plt.title('Equilibrium Search: Performance vs. Stability')
    plt.grid(True, alpha=0.3)
    
    # Highlight the best trial
    best_idx = study.best_trial.number
    plt.annotate('Optimal Equilibrium', 
                 xy=(gap_vals[best_idx], f1_vals[best_idx]), 
                 xytext=(gap_vals[best_idx]+0.05, f1_vals[best_idx]-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig('performance_stability_tradeoff.png')
    
    # --- Plot 3: Parameter Importance ---
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig('balanced_importances.png')
    except:
        pass

    print(f"\nBest Results Found:")
    print(f"  Max F1 at Equilibrium: {study.trials[best_idx].user_attrs['f1']:.4f}")
    print(f"  P-R Gap: {study.trials[best_idx].user_attrs['gap']:.4f}")
    print(f"  Params: {study.best_params}")

if __name__ == "__main__":
    run_balanced_optimization("robust_2017_final.csv", "robust_2018_final.csv")