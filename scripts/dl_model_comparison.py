import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score, precision_recall_curve

# --- 1. Model Definitions ---

# Current Optimized MLP (32, 16)
class MLP_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# CNN (1D Convolution for feature extraction)
class CNN_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Unflatten(1, (1, input_dim)),
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * input_dim, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.conv(x)

# LSTM (Recurrent approach for temporal features)
class LSTM_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(1))
        return self.sig(self.fc(x[:, -1, :]))

# Autoencoder-based Classifier (Anomaly Detection style)
class AE_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(16, input_dim), nn.Sigmoid())
        self.clf = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        latent = self.encoder(x)
        return self.sig(self.clf(latent))

# --- 2. Evaluation Wrapper ---

def evaluate_model(model_class, X_train, y_train, X_test, y_test):
    model = model_class(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    start_time = time.time()
    model.train()
    for epoch in range(10):
        for bx, by in loader:
            optimizer.zero_grad(); criterion(model(bx), by).backward(); optimizer.step()
    
    train_time = time.time() - start_time
    
    # Eval with Optimized Threshold (Simulating AQT)
    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy()
        y_true = y_test.numpy()
        p, r, t = precision_recall_curve(y_true, probs)
        f1 = (2 * p * r) / (p + r + 1e-8)
        best_f1 = np.max(f1)
        
    return best_f1, train_time

# --- 3. Execution ---

def run_comparison(train_csv, test_csv):
    train_df = pd.read_csv(train_csv).sample(1000)
    test_df = pd.read_csv(test_csv)
    
    X_tr = torch.tensor(train_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_tr = torch.tensor(train_df['Label'].values, dtype=torch.float32).reshape(-1, 1)
    X_te = torch.tensor(test_df.drop(columns=['Label']).values, dtype=torch.float32)
    y_te = torch.tensor(test_df['Label'].values, dtype=torch.float32).reshape(-1, 1)

    models = {"MLP (Ours)": MLP_Model, "CNN": CNN_Model, "LSTM": LSTM_Model, "Autoencoder": AE_Model}
    results = []

    for name, m_class in models.items():
        f1, t_time = evaluate_model(m_class, X_tr, y_tr, X_te, y_te)
        results.append({"Model": name, "F1": f1, "TrainTime": t_time})
        print(f"{name:12} | F1: {f1:.4f} | Time: {t_time:.4f}s")

    # --- Plotting Results ---
    res_df = pd.DataFrame(results)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Model Architecture')
    ax1.set_ylabel('F1-Score', color='blue')
    ax1.bar(res_df['Model'], res_df['F1'], color='blue', alpha=0.6, label='F1 Performance')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Training Time (s)', color='red')
    ax2.plot(res_df['Model'], res_df['TrainTime'], color='red', marker='D', linewidth=2, label='Training Latency')

    plt.title('DL Architecture Comparison: Accuracy vs. Computational Overhead')
    plt.savefig('dl_model_comparison.png')
    plt.show()

if __name__ == "__main__":
    run_comparison("robust_2017_final.csv", "robust_2018_final.csv")