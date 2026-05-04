import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import copy
import warnings

warnings.filterwarnings("ignore")

def find_top_22_features(df_cols):
    keywords = [
        'Init_Fwd_Win', 'Init_Bwd_Win', 'Fwd_Pkt_Len_Std', 'Bwd_Pkt_Len_Max',
        'Packet_Len_Var', 'Flow_IAT_Min', 'Bwd_IAT_Max', 'Flow_Duration',
        'Flow_Pkts', 'Flow_Byts', 'Fwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Std',
        'Flow_IAT_Mean', 'Fwd_IAT_Max', 'Bwd_Pkt_Len_Min', 'Fwd_Header',
        'Bwd_Header', 'Fwd_IAT_Tot', 'Bwd_IAT_Tot', 'Active_Max', 'Idle_Max', 'Subflow_Fwd'
    ]
    found_cols = []
    for kw in keywords:
        match = [c for c in df_cols if kw.replace('_', '').lower() in c.replace('_', '').replace(' ', '').lower()]
        if match: found_cols.append(match[0])
    return list(set(found_cols))

def get_actual_attack_label(df, target_name):
    """Fuzzy matches the attack name in the Label column to handle typos/casing."""
    unique_labels = df['Label'].unique()
    for label in unique_labels:
        if target_name.lower() in str(label).lower():
            return label
    return None

def prepare_balanced_source(df, attack_type, features, total_samples=5000):
    # Find the real label string in the 2017 data
    real_label = get_actual_attack_label(df, attack_type)
    
    if real_label is None:
        raise ValueError(f"Attack type '{attack_type}' not found in 2017 dataset. Available: {df['Label'].unique()}")
        
    benign_df = df[df['Label'] == 'Benign']
    attack_df = df[df['Label'] == real_label]
    
    # Check if we actually have attack data to sample
    if len(attack_df) == 0:
        raise ValueError(f"No samples found for {real_label} in source data.")

    n_each = total_samples // 2
    s_benign = benign_df.sample(n=n_each, random_state=42, replace=True)
    s_attack = attack_df.sample(n=n_each, random_state=42, replace=True)
    
    balanced_df = pd.concat([s_benign, s_attack])
    return shuffle(balanced_df, random_state=42), real_label

def run_experiment(df_17, df_18, attack_name, features):
    test_sizes = [100, 500, 1000, 2500, 5000, 7200, 10000]
    n_iterations = 10
    
    # Pre-train Phase (2017)
    df_src, src_label_str = prepare_balanced_source(df_17, attack_name, features)
    X_src = df_src[features]
    y_src = df_src['Label'].apply(lambda x: 1 if x == src_label_str else 0)
    
    base_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    base_mlp.fit(X_src, y_src)
    
    # Prep Target (2018)
    tar_label_str = get_actual_attack_label(df_18, attack_name)
    df_target_full = df_18[(df_18['Label'] == 'Benign') | (df_18['Label'] == tar_label_str)]
    df_target_full['Binary_Label'] = df_target_full['Label'].apply(lambda x: 1 if x == tar_label_str else 0)
    
    df_test = df_target_full.sample(frac=0.2, random_state=99)
    df_adapt_pool = df_target_full.drop(df_test.index)
    X_test = df_test[features]
    y_test = df_test['Binary_Label']

    tl_means = []
    for size in test_sizes:
        iter_results = []
        for i in range(n_iterations):
            # Ensure we don't try to sample more than available in the pool
            current_size = min(size, len(df_adapt_pool))
            df_adapt = df_adapt_pool.sample(n=current_size, random_state=i+size)
            tl_model = copy.deepcopy(base_mlp)
            tl_model.partial_fit(df_adapt[features], df_adapt['Binary_Label'])
            iter_results.append(f1_score(y_test, tl_model.predict(X_test)))
        tl_means.append(np.mean(iter_results))
        
    return test_sizes, tl_means

# --- EXECUTION ---
try:
    #df_2017 = pd.read_csv('robust_2017_final.csv')
    #df_2018 = pd.read_csv('robust_2018_final.csv')
    df_2017 = pd.read_csv('master_stratified_dataset_2017.csv')
    df_2018 = pd.read_csv('master_stratified_dataset_2018.csv')

    features = find_top_22_features(df_2017.columns)
    print(f"Mapped {len(features)}/22 features.")

    # These are general names; the fuzzy matcher will find 'Infiltration', 'Infilteration', etc.
    attacks = ['Infiltration', 'Botnet', 'DDoS']
    plt.figure(figsize=(10, 6))

    for attack in attacks:
        try:
            print(f"Analyzing {attack}...")
            sizes, f1_scores = run_experiment(df_2017, df_2018, attack, features)
            plt.plot(sizes, f1_scores, marker='o', label=f'{attack} (22-Feat TL)')
        except ValueError as ve:
            print(f"Skipping {attack}: {ve}")

    plt.axvline(x=7200, color='red', linestyle='--', label='7,200 Reliability Threshold')
    plt.title('Monte Carlo Convergence: 22-Feature Real Data Transfer')
    plt.xlabel('Target Samples (2018)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('top_22_real_data_convergence.png')
    plt.show()

except Exception as e:
    print(f"Critical Error: {e}")
    