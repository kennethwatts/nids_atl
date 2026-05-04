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

# EXACT TOP 22 FEATURES FROM YOUR COLUMN LIST
top_22_features = [
    'Init_Fwd_Win_Byts', 'Bwd_Init_Win_Bytes', 'Fwd_Pkt_Len_Std', 
    'Bwd_Pkt_Len_Max', 'Pkt_Len_Var', 'Flow_IAT_Min', 
    'Bwd_IAT_Max', 'Flow_Duration', 'Flow_Pkts_s', 
    'Flow_Bytes_s', 'Fwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Std',
    'Flow_IAT_Mean', 'Fwd_IAT_Max', 'Bwd_Pkt_Len_Min',
    'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_IAT_Tot',
    'Bwd_IAT_Tot', 'Active_Max', 'Idle_Max', 'Subflow_Fwd_Bytes'
]

def clean_and_scale(df_17, df_18):
    """Cleans NaNs, handles BOM/string issues, and normalizes data."""
    for df in [df_17, df_18]:
        # Strip any leading/trailing whitespace from column names (common with BOM)
        df.columns = df.columns.str.strip()
        
        # Force conversion to numeric, turning errors like '\ufeff67' into NaN
        for col in top_22_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Standard cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=top_22_features + ['Label'], inplace=True)
    
    scaler = StandardScaler()
    # Fit on 2017 baseline, apply to both to preserve domain shift
    df_17[top_22_features] = scaler.fit_transform(df_17[top_22_features])
    df_18[top_22_features] = scaler.transform(df_18[top_22_features])
    return df_17, df_18

def run_experiment(df_17, df_18, attack_name):
    test_sizes = [100, 500, 1000, 2500, 5000, 7200, 10000]
    n_iterations = 10 
    
    # Stratified Pre-train on 2017
    src_data = df_17[df_17['Label'].isin(['BENIGN', attack_name])].copy()
    src_data['y'] = src_data['Label'].apply(lambda x: 1 if x == attack_name else 0)
    
    n_base = 2000 
    s_benign = src_data[src_data['y']==0].sample(n=n_base//2, replace=True, random_state=42)
    s_attack = src_data[src_data['y']==1].sample(n=n_base//2, replace=True, random_state=42)
    train_df = shuffle(pd.concat([s_benign, s_attack]), random_state=42)
    
    base_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, alpha=1e-5, random_state=42)
    base_mlp.fit(train_df[top_22_features], train_df['y'])
    
    # Target Adaptation (2018)
    tar_data = df_18[df_18['Label'].isin(['BENIGN', attack_name])].copy()
    tar_data['y'] = tar_data['Label'].apply(lambda x: 1 if x == attack_name else 0)
    
    test_df = tar_data.sample(n=min(1500, len(tar_data)//2), random_state=99)
    adapt_pool = tar_data.drop(test_df.index)
    
    tl_means = []
    for size in test_sizes:
        iter_results = []
        actual_size = min(size, len(adapt_pool))
        for i in range(n_iterations):
            batch = adapt_pool.sample(n=actual_size, random_state=i+size)
            tl_model = copy.deepcopy(base_mlp)
            tl_model.partial_fit(batch[top_22_features], batch['y'], classes=[0, 1])
            iter_results.append(f1_score(test_df['y'], tl_model.predict(test_df[top_22_features])))
        tl_means.append(np.mean(iter_results))
        
    return test_sizes, tl_means

# --- EXECUTION ---
try:
    print("Loading datasets (UTF-8-SIG to strip BOM)...")
    # Using utf-8-sig to automatically handle the \ufeff characters
    df17 = pd.read_csv('master_stratified_dataset_2017.csv', encoding='utf-8-sig', low_memory=False)
    df18 = pd.read_csv('master_stratified_dataset_2018.csv', encoding='utf-8-sig', low_memory=False)
    
    print("Cleaning and Scaling Data...")
    df17, df18 = clean_and_scale(df17, df18)

    attacks = ['Infiltration', 'Bot', 'DDoS'] 
    plt.figure(figsize=(10, 6))

    for attack in attacks:
        print(f"Running Monte Carlo for {attack}...")
        try:
            sizes, scores = run_experiment(df17, df18, attack)
            plt.plot(sizes, scores, marker='o', label=f"{attack} (BOM-Cleaned TL)")
        except Exception as e:
            print(f"Skipping {attack}: {e}")

    plt.axvline(x=7200, color='red', linestyle='--', label='7,200 Reliability Threshold')
    plt.title('Monte Carlo Convergence: Adaptive TL (BOM-Cleaned Real Data)')
    plt.xlabel('Adaptation Samples (2018)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('final_top_22_bom_safe_convergence.png')
    plt.show()

except Exception as e:
    print(f"Critical Failure: {e}")