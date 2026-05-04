import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import copy
import warnings

# Suppress ConvergenceWarnings for small early batches
warnings.filterwarnings("ignore")

def prepare_balanced_source(df, attack_type, total_samples=5000):
    """
    Enforces a 1:1 ratio between Benign and Attack classes in the 2017 source data.
    This prevents 'Benign Bias' and ensures a robust behavioral pre-training.
    """
    benign_df = df[df['Label'] == 'Benign']
    attack_df = df[df['Label'] == attack_type]
    
    n_each = total_samples // 2
    
    # Stratified sampling with replacement if necessary
    s_benign = benign_df.sample(n=n_each, random_state=42, replace=True)
    s_attack = attack_df.sample(n=n_each, random_state=42, replace=True)
    
    balanced_df = pd.concat([s_benign, s_attack])
    return shuffle(balanced_df, random_state=42)

def run_monte_carlo_experiment(df_17, df_18, attack_name):
    """
    Runs the 10-iteration Monte Carlo simulation to evaluate F1 convergence.
    Compares Adaptive TL against Training From Scratch.
    """
    test_sizes = [100, 500, 1000, 2500, 5000, 7200, 10000]
    n_iterations = 10
    
    # PHASE 1: PRE-TRAINING (Source Domain 2017)
    # Focuses the base weights on the core protocol behavioral manifold
    df_src = prepare_balanced_source(df_17, attack_name, total_samples=5000)
    X_src = df_src.drop('Label', axis=1)
    y_src = df_src['Label'].apply(lambda x: 1 if x == attack_name else 0)
    
    base_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    base_mlp.fit(X_src, y_src)
    
    # PHASE 2: PREPARE TARGET (2018)
    # Filter target domain for the relevant attack class
    df_target_full = df_18[(df_18['Label'] == 'Benign') | (df_18['Label'] == attack_name)]
    df_target_full['Binary_Label'] = df_target_full['Label'].apply(lambda x: 1 if x == attack_name else 0)
    
    # Hold out fixed test set (20% of target sub-domain)
    df_test = df_target_full.sample(frac=0.2, random_state=99)
    df_adapt_pool = df_target_full.drop(df_test.index)
    
    X_test = df_test.drop(['Label', 'Binary_Label'], axis=1)
    y_test = df_test['Binary_Label']

    # Containers for results
    tl_means = []
    scratch_means = []

    for size in test_sizes:
        iter_tl = []
        iter_scratch = []
        
        for i in range(n_iterations):
            # Sample current adaptation batch from target domain
            df_adapt = df_adapt_pool.sample(n=size, random_state=i+size)
            X_adapt = df_adapt.drop(['Label', 'Binary_Label'], axis=1)
            y_adapt = df_adapt['Binary_Label']
            
            # 1. Adaptive Transfer Learning
            # Fine-tunes the 2017 pre-trained weights to 2018 infrastructure
            tl_model = copy.deepcopy(base_mlp)
            tl_model.partial_fit(X_adapt, y_adapt)
            iter_tl.append(f1_score(y_test, tl_model.predict(X_test)))
            
            # 2. Training From Scratch (Baseline)
            # Starts with random weights and only sees target data
            scratch_model = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=42)
            scratch_model.partial_fit(X_adapt, y_adapt, classes=[0, 1])
            iter_scratch.append(f1_score(y_test, scratch_model.predict(X_test)))
            
        tl_means.append(np.mean(iter_tl))
        scratch_means.append(np.mean(iter_scratch))
        
    return test_sizes, tl_means, scratch_means

# --- EXECUTION ---
# Ensure your 14-feature robust CSVs are in the working directory
try:
    #master_stratified_dataset_2018
    #df_2017 = pd.read_csv('robust_2017_final.csv')
    #df_2018 = pd.read_csv('robust_2018_final.csv')
    df_2017 = pd.read_csv('master_stratified_dataset_2017.csv')
    df_2018 = pd.read_csv('master_stratified_dataset_2018.csv')

    # Define the 22 features identified in your PCA analysis
    # (Ensure these match the exact column names in your CSVs)
    top_22_features = [
        'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts', 'Fwd_Pkt_Len_Std', 
        'Bwd_Pkt_Len_Max', 'Packet_Len_Var', 'Flow_IAT_Min', 
        'Bwd_IAT_Max', 'Flow_Duration', 'Flow_Pkts_s', 
        'Flow_Byts_s', 'Fwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Std',
        'Flow_IAT_Mean', 'Fwd_IAT_Max', 'Bwd_Pkt_Len_Min',
        'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_IAT_Total',
        'Bwd_IAT_Total', 'Active_Max', 'Idle_Max', 'Subflow_Fwd_Byts'
    ]

    # Apply filtering to your robust CSVs before running the MLP
    # We keep 'Label' for the stratification and training process
    df_2017_22 = df_2017[top_22_features + ['Label']]
    df_2018_22 = df_2018[top_22_features + ['Label']]

    print(f"New Feature Space Shape: {df_2017_22.shape[1] - 1} features + Label")

    attacks = ['Infiltration', 'Botnet', 'DDoS']
    plt.figure(figsize=(12, 7))

    for attack in attacks:
        print(f"Running Monte Carlo for: {attack}")
        sizes, tl_f1, scratch_f1 = run_monte_carlo_experiment(df_2017_22, df_2018_22, attack)
        
        plt.plot(sizes, tl_f1, marker='o', label=f'{attack} (Adaptive TL)')
        plt.plot(sizes, scratch_f1, linestyle='--', alpha=0.5, label=f'{attack} (Scratch)')

    plt.axvline(x=7200, color='red', linestyle=':', label='7,200 Reliability Threshold')
    plt.title('Monte Carlo Convergence: Adaptive TL vs. From-Scratch (Real Data)')
    plt.xlabel('Target Adaptation Samples (CSE-CIC-IDS2018)')
    plt.ylabel('F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('final_real_data_convergence.png')
    plt.show()

except FileNotFoundError:
    print("Error: robust_2017_final.csv or robust_2018_final.csv not found.")
    