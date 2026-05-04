import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import copy
import warnings

warnings.filterwarnings("ignore")

# Verified Top 22 Features
top_22_features = [
    'Init_Fwd_Win_Byts', 'Bwd_Init_Win_Bytes', 'Fwd_Pkt_Len_Std', 
    'Bwd_Pkt_Len_Max', 'Pkt_Len_Var', 'Flow_IAT_Min', 
    'Bwd_IAT_Max', 'Flow_Duration', 'Flow_Pkts_s', 
    'Flow_Bytes_s', 'Fwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Std',
    'Flow_IAT_Mean', 'Fwd_IAT_Max', 'Bwd_Pkt_Len_Min',
    'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_IAT_Tot',
    'Bwd_IAT_Tot', 'Active_Max', 'Idle_Max', 'Subflow_Fwd_Bytes'
]

def clean_and_binary(df):
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in top_22_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=top_22_features + ['Label'], inplace=True)
    df['y'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    return df

# --- 1. Load and Standardize ---
print("Loading and Scaling Data...")
df17 = clean_and_binary(pd.read_csv('master_stratified_dataset_2017.csv', encoding='utf-8-sig', low_memory=False))
df18 = clean_and_binary(pd.read_csv('master_stratified_dataset_2018.csv', encoding='utf-8-sig', low_memory=False))

scaler = StandardScaler()
df17[top_22_features] = scaler.fit_transform(df17[top_22_features])
df18[top_22_features] = scaler.transform(df18[top_22_features])

# --- 2. Pre-train 2017 Base ---
s0 = df17[df17['y']==0].sample(5000, replace=True, random_state=42)
s1 = df17[df17['y']==1].sample(5000, replace=True, random_state=42)
base_train = shuffle(pd.concat([s0, s1]), random_state=42)

base_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1, warm_start=True, random_state=42)
base_model.fit(base_train[top_22_features], base_train['y'])

# --- 3. Loss Comparison Loop ---
# We use a fixed batch size to see the step-by-step "Catch-up"
adaptation_steps = np.arange(1000, 51000, 2000)
tl_loss, ml_loss = [], []

pool_df = shuffle(df18, random_state=42)

for size in adaptation_steps:
    batch = pool_df.sample(n=size, random_state=42)
    
    # Adaptive TL
    tl_mod = copy.deepcopy(base_model)
    tl_mod.partial_fit(batch[top_22_features], batch['y'])
    tl_loss.append(tl_mod.loss_)
    
    # Traditional ML
    ml_mod = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=42)
    ml_mod.partial_fit(batch[top_22_features], batch['y'], classes=[0, 1])
    ml_loss.append(ml_mod.loss_)

# --- 4. Plotting the "Convergence Gap" ---
plt.figure(figsize=(10, 6))
plt.plot(adaptation_steps, tl_loss, 'b-', label='Adaptive TL Loss (Pre-trained)')
plt.plot(adaptation_steps, ml_loss, 'r--', label='Traditional ML Loss (Scratch)')
plt.title('Training Loss Convergence: Does Traditional ML Catch Up?')
plt.xlabel('2018 Training Samples')
plt.ylabel('Log-Loss (Lower is better)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_convergence_comparison.png')
plt.show()