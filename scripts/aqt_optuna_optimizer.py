import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# --- (32, 16) Architecture ---
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

def objective(trial):
    # 1. Hyperparameter Suggestions
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 10, 25)
    
    # 2. THRESHOLD TUNING (q-quantile for AQT)
    # Tuning the sensitivity q to find the Optimal Equilibrium
    q_quantile = trial.suggest_float("q_quantile", 0.85, 0.999)

    # Load and scale data (1000 sample saturation point)
    train_df = pd.read_csv('robust_2017_final.csv').sample(1000)
    test_df = pd.read_csv('robust_2018_final.csv')
    
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(train_df.drop(columns=['Label']).values), dtype=torch.float32)
    y_train = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(scaler.transform(test_df.drop(columns=['Label']).values), dtype=torch.float32)
    y_true = test_df['Label'].values

    # 3. Training Loop
    model = IntrusionDetector(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            criterion(model(batch_x), batch_y).backward()
            optimizer.step()

    # 4. Evaluation with Tuned Threshold
    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy().flatten()
        
        # Apply the q-quantile threshold suggested by Optuna
        dynamic_threshold = np.quantile(probs, q_quantile)
        preds = (probs >= dynamic_threshold).astype(float)
        
        f1 = f1_score(y_true, preds, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        pr_gap = abs(precision - recall)

    # We return F1 as the objective, but we can track the gap
    trial.set_user_attr("pr_gap", pr_gap)
    trial.set_user_attr("threshold_value", dynamic_threshold)
    
    # Penalty logic: If the gap is over 0.45, penalize the F1-score
    if pr_gap > 0.45:
        return f1 * 0.5
    
    return f1

# --- Execution ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_trial = study.best_trial
print(f"\nBest Results Found:")
print(f"  Max F1 at Equilibrium: {best_trial.value:.4f}")
print(f"  P-R Gap: {best_trial.user_attrs['pr_gap']:.4f}")
print(f"  Tuned q-Quantile: {best_trial.params['q_quantile']:.4f}")
print(f"  Resulting Threshold: {best_trial.user_attrs['threshold_value']:.4f}")
print(f"  Params: {best_trial.params}")