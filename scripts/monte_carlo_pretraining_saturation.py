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
print("Loading datasets...")
df17 = clean_and_binary(pd.read_csv('master_stratified_dataset_2017.csv', encoding='utf-8-sig', low_memory=False))
df18 = clean_and_binary(pd.read_csv('master_stratified_dataset_2018.csv', encoding='utf-8-sig', low_memory=False))

scaler = StandardScaler()
df17[top_22_features] = scaler.fit_transform(df17[top_22_features])
df18[top_22_features] = scaler.transform(df18[top_22_features])

# --- 2. Monte Carlo Experiment Loop ---
# Ranges from 1k to 10k (fine grain) then 20k to 100k (coarse grain)
pretrain_sizes = [1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
mc_iterations = 50 
avg_f1_scores = []
std_f1_scores = []

# Fixed test set from 2018
test_df = df18.sample(n=min(2000, len(df18)//4), random_state=99)

for size in pretrain_sizes:
    print(f"Analyzing Pre-training Volume: {size}...")
    iteration_scores = []
    
    for i in range(mc_iterations):
        # Stratified sample from 2017 with varying seeds
        s0 = df17[df17['y']==0].sample(size // 2, replace=True, random_state=i*size)
        s1 = df17[df17['y']==1].sample(size // 2, replace=True, random_state=i+size)
        train_df = shuffle(pd.concat([s0, s1]), random_state=i)
        
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, alpha=1e-5, random_state=42)
        model.fit(train_df[top_22_features], train_df['y'])
        
        score = f1_score(test_df['y'], model.predict(test_df[top_22_features]))
        iteration_scores.append(score)
        
    avg_f1_scores.append(np.mean(iteration_scores))
    std_f1_scores.append(np.std(iteration_scores))

# --- 3. Plotting the Flattened Graph ---
plt.figure(figsize=(10, 6))
plt.errorbar(pretrain_sizes, avg_f1_scores, yerr=std_f1_scores, fmt='-o', 
             color='blue', ecolor='lightblue', elinewidth=3, capsize=0, 
             label='Mean F1 (50 MC Iterations)')

plt.title('Source Volume Impact: Monte Carlo Saturation Analysis (2017 to 2018)')
plt.xlabel('Number of Pre-training Samples (CICIDS2017)')
plt.ylabel('Initial F1 Score on 2018 Data')
plt.xscale('log') # Log scale helps visualize the "elbow" better
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.savefig('monte_carlo_pretraining_saturation.png')
plt.show()