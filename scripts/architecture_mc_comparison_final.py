import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import warnings
import copy

warnings.filterwarnings("ignore")

# --- Configuration & Features ---
TOP_22_FEATURES = [
    'Init_Fwd_Win_Byts', 'Bwd_Init_Win_Bytes', 'Fwd_Pkt_Len_Std', 
    'Bwd_Pkt_Len_Max', 'Pkt_Len_Var', 'Flow_IAT_Min', 
    'Bwd_IAT_Max', 'Flow_Duration', 'Flow_Pkts_s', 
    'Flow_Bytes_s', 'Fwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Std',
    'Flow_IAT_Mean', 'Fwd_IAT_Max', 'Bwd_Pkt_Len_Min',
    'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_IAT_Tot',
    'Bwd_IAT_Tot', 'Active_Max', 'Idle_Max', 'Subflow_Fwd_Bytes'
]

def clean_and_binary(df):
    """Sanitizes dataset and creates binary labels (0: Benign, 1: Attack)."""
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in TOP_22_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=TOP_22_FEATURES + ['Label'], inplace=True)
    # Generalist approach: BENIGN is 0, all other attacks are 1
    df['y'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    return df

# --- 1. Data Loading & Preprocessing ---
print("Loading and Sanitizing Master Datasets...")
try:
    df17 = clean_and_binary(pd.read_csv('master_stratified_dataset_2017.csv', encoding='utf-8-sig', low_memory=False))
    df18 = clean_and_binary(pd.read_csv('master_stratified_dataset_2018.csv', encoding='utf-8-sig', low_memory=False))

    scaler = StandardScaler()
    df17[TOP_22_FEATURES] = scaler.fit_transform(df17[TOP_22_FEATURES])
    df18[TOP_22_FEATURES] = scaler.transform(df18[TOP_22_FEATURES])
except FileNotFoundError as e:
    print(f"Error: Ensure the dataset files are in the local directory. {e}")
    exit()

# --- 2. Monte Carlo Setup ---
pretrain_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
mc_iterations = 50  # Balanced for statistical smoothing and execution time
test_df = df18.sample(n=min(2000, len(df18)//4), random_state=99)

architectures = {
    'Lightweight (32, 16)': (32, 16),
    'High-Capacity (256, 128)': (256, 128)
}

results = {name: [] for name in architectures.keys()}
variances = {name: [] for name in architectures.keys()}

# --- 3. Execution Loop ---
for arch_name, shape in architectures.items():
    print(f"\nStarting Monte Carlo Analysis for {arch_name}...")
    
    for size in pretrain_sizes:
        iteration_scores = []
        print(f"  Processing Pre-training Volume: {size} samples", end=" | ")
        
        for i in range(mc_iterations):
            # Balanced stratified sampling from 2017
            s0 = df17[df17['y']==0].sample(size // 2, replace=True, random_state=i*size)
            s1 = df17[df17['y']==1].sample(size // 2, replace=True, random_state=i+size)
            train_df = shuffle(pd.concat([s0, s1]), random_state=i)
            
            # MLP training
            model = MLPClassifier(hidden_layer_sizes=shape, max_iter=500, alpha=1e-5, random_state=42)
            model.fit(train_df[TOP_22_FEATURES], train_df['y'])
            
            # Predict on 2018 (Zero local adaptation)
            preds = model.predict(test_df[TOP_22_FEATURES])
            iteration_scores.append(f1_score(test_df['y'], preds))
        
        avg_f1 = np.mean(iteration_scores)
        std_f1 = np.std(iteration_scores)
        results[arch_name].append(avg_f1)
        variances[arch_name].append(std_f1)
        print(f"Avg F1: {avg_f1:.4f}")

# --- 4. Plotting Results ---
plt.figure(figsize=(12, 7))

colors = {'Lightweight (32, 16)': 'blue', 'High-Capacity (256, 128)': 'green'}
markers = {'Lightweight (32, 16)': 'o', 'High-Capacity (256, 128)': '^'}

for arch_name in architectures.keys():
    plt.errorbar(pretrain_sizes, results[arch_name], yerr=variances[arch_name], 
                 fmt=f'-{markers[arch_name]}', color=colors[arch_name], 
                 elinewidth=1.5, capsize=3, label=arch_name, linewidth=2)

plt.xscale('log')
plt.title('Impact of Network Architecture on Cross-Domain F1 Transfer Ceiling')
plt.xlabel('Number of Pre-training Samples (CICIDS2017)')
plt.ylabel('Initial F1 Score on 2018 Data')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower right')

# Add context line for the 0.60 ceiling
plt.axhline(y=0.60, color='red', linestyle='--', alpha=0.5, label='Expected Domain Ceiling')

plt.tight_layout()
plt.savefig('architecture_mc_comparison_final.png', dpi=300)
plt.show()

print("\nAnalysis Complete. Graph saved as 'architecture_mc_comparison_final.png'.")