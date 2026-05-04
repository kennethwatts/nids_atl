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

def clean_and_binary_label(df):
    """Cleans data and converts all attacks to class 1."""
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Force features to numeric
    for col in top_22_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df.dropna(subset=top_22_features + ['Label'], inplace=True)
    
    # Binary Labeling: BENIGN is 0, everything else is 1
    df['y'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    return df

# --- 1. Load and Prepare ---
try:
    print("Loading and Sanitizing Master Datasets...")
    # Updated to correct function name
    df17 = clean_and_binary_label(pd.read_csv('master_stratified_dataset_2017.csv', encoding='utf-8-sig', low_memory=False))
    df18 = clean_and_binary_label(pd.read_csv('master_stratified_dataset_2018.csv', encoding='utf-8-sig', low_memory=False))

    scaler = StandardScaler()
    df17[top_22_features] = scaler.fit_transform(df17[top_22_features])
    df18[top_22_features] = scaler.transform(df18[top_22_features])

    # --- 2. Phase 1: Pre-training (The 2017 Base) ---
    print("Pre-training Generalist Model on 2017 Manifold...")
    s0 = df17[df17['y']==0].sample(2500, random_state=42, replace=True)
    s1 = df17[df17['y']==1].sample(2500, random_state=42, replace=True)
    base_train = shuffle(pd.concat([s0, s1]), random_state=42)

    base_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, alpha=1e-5, random_state=42)
    base_model.fit(base_train[top_22_features], base_train['y'])

    # --- 3. Phase 2: Monte Carlo Comparison ---
    test_sizes = [100, 500, 1000, 2500, 5000, 7200, 10000]
    test_df = df18.sample(n=min(2000, len(df18)//4), random_state=99)
    pool_df = df18.drop(test_df.index)

    tl_results, scratch_results = [], []

    for size in test_sizes:
        print(f"Testing sample size: {size}...")
        it_tl, it_scratch = [], []
        actual_size = min(size, len(pool_df))
        
        for i in range(50):
            batch = pool_df.sample(actual_size, random_state=i+size)
            
            # Adaptive Transfer Learning
            tl_mod = copy.deepcopy(base_model)
            tl_mod.partial_fit(batch[top_22_features], batch['y'])
            it_tl.append(f1_score(test_df['y'], tl_mod.predict(test_df[top_22_features])))
            
            # Traditional ML (Training From Scratch)
            scratch_mod = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=42)
            scratch_mod.partial_fit(batch[top_22_features], batch['y'], classes=[0, 1])
            it_scratch.append(f1_score(test_df['y'], scratch_mod.predict(test_df[top_22_features])))
            
        tl_results.append(np.mean(it_tl))
        scratch_results.append(np.mean(it_scratch))

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, tl_results, 'b-o', label='Adaptive Transfer Learning (Pre-trained)')
    plt.plot(test_sizes, scratch_results, 'r--x', label='Traditional ML (From Scratch)')
    plt.axvline(x=7200, color='gray', linestyle=':', label='Reliability Threshold')
    plt.title('Generalist Detector: Transfer Learning vs. Traditional ML')
    plt.xlabel('2018 Adaptation Samples')
    plt.ylabel('Aggregate F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('generalist_comparison.png')
    plt.show()

except Exception as e:
    print(f"Critical Error: {e}")