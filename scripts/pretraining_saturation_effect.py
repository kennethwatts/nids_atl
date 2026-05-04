import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
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
print("Loading datasets...")
df17 = clean_and_binary(pd.read_csv('master_stratified_dataset_2017.csv', encoding='utf-8-sig', low_memory=False))
df18 = clean_and_binary(pd.read_csv('master_stratified_dataset_2018.csv', encoding='utf-8-sig', low_memory=False))

scaler = StandardScaler()
df17[top_22_features] = scaler.fit_transform(df17[top_22_features])
df18[top_22_features] = scaler.transform(df18[top_22_features])

# --- 2. Experiment Loop (High Volume) ---
pretrain_sizes = np.arange(10000, 110000, 10000)
initial_f1_scores = []

# Fixed test set from 2018
test_df = df18.sample(n=min(2000, len(df18)//4), random_state=99)

for size in pretrain_sizes:
    print(f"Pre-training with {size} samples from 2017...")
    
    # Stratified sample from 2017
    # Note: Using replace=True in case any attack class has fewer than 50k samples
    s0 = df17[df17['y']==0].sample(size // 2, replace=True, random_state=42)
    s1 = df17[df17['y']==1].sample(size // 2, replace=True, random_state=42)
    train_df = shuffle(pd.concat([s0, s1]), random_state=42)
    
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, alpha=1e-5, random_state=42)
    model.fit(train_df[top_22_features], train_df['y'])
    
    # Test immediately on 2018
    preds = model.predict(test_df[top_22_features])
    score = f1_score(test_df['y'], preds)
    initial_f1_scores.append(score)
    print(f"-> Initial F1 Score on 2018: {score:.4f}")

# --- 3. Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(pretrain_sizes, initial_f1_scores, 'b-o', linewidth=2, markersize=8)
plt.title('Source Volume Saturation: 10k-100k 2017 Pre-training vs. Initial 2018 F1')
plt.xlabel('Number of Pre-training Samples (CICIDS2017)')
plt.ylabel('Initial F1 Score (CSE-CIC-IDS2018)')
plt.grid(True, alpha=0.3)
plt.xticks(pretrain_sizes)

plt.savefig('pretraining_saturation_effect.png')
plt.show()