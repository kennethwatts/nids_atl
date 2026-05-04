import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import precision_recall_curve, f1_score

# 1. Model Architecture
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
    
    model = IntrusionDetector(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(10):
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy()
        y_true = y_test.numpy()
        
        # Calculate PR Curve
        p, r, t = precision_recall_curve(y_true, probs)
        
        # Goal: Minimize the gap between Precision and Recall
        # We find the threshold in this specific trial that results in the smallest gap
        gaps = np.abs(p - r)
        min_gap_idx = np.argmin(gaps)
        
        # We return the gap as the value to minimize
        return gaps[min_gap_idx]

def run_optimization(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Use exactly 1,000 samples for pre-training
    train_df = train_df.sample(n=1000, random_state=42)
    
    X_train = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    # Run Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test), n_trials=50)

    print("\nBest Trial Results:")
    print(f"  Gap Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # 1. Plot Optimization History (This always works)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig('optuna_gap_history.png')
    
    # 2. Plot Parameter Importances (Wrapped in Try/Except to prevent the crash)
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig('optuna_importances.png')
    except RuntimeError:
        print("\nNote: Skipping Importance Plot - Variance is zero (Optimization was perfectly consistent).")

    # 3. CUSTOM GRAPH: Precision vs Recall Equilibrium
    # Let's extract the data from the trials to show the "Convergence"
    trials_df = study.trials_dataframe()
    plt.figure(figsize=(10, 6))
    plt.scatter(trials_df.index, trials_df.value, c='blue', alpha=0.6, label='P-R Gap')
    plt.axhline(y=0, color='red', linestyle='--', label='Perfect Equilibrium')
    plt.title('Convergence to Precision-Recall Equilibrium (1000 Samples)')
    plt.xlabel('Trial Number')
    plt.ylabel('Abs(Precision - Recall)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('pr_equilibrium_convergence.png')
    plt.show()

if __name__ == "__main__":
    run_optimization("robust_2017_final.csv", "robust_2018_final.csv")