import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import copy
import warnings

warnings.filterwarnings("ignore")

# Verified Top 22 Features
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
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in TOP_22_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=TOP_22_FEATURES + ['Label'], inplace=True)
    df['y'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    return df

# --- 1. Load and Prepare ---
print("Loading and Scaling Data...")
df17 = clean_and_binary(pd.read_csv('master_stratified_dataset_2017.csv', encoding='utf-8-sig', low_memory=False))
df18 = clean_and_binary(pd.read_csv('master_stratified_dataset_2018.csv', encoding='utf-8-sig', low_memory=False))

scaler = StandardScaler()
df17[TOP_22_FEATURES] = scaler.fit_transform(df17[TOP_22_FEATURES])
df18[TOP_22_FEATURES] = scaler.transform(df18[TOP_22_FEATURES])

# --- 2. Pre-train 2017 Base (Adaptive TL Start) ---
s0 = df17[df17['y']==0].sample(5000, replace=True, random_state=42)
s1 = df17[df17['y']==1].sample(5000, replace=True, random_state=42)
base_train = shuffle(pd.concat([s0, s1]), random_state=42)

base_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1, warm_start=True, random_state=42)
base_model.fit(base_train[TOP_22_FEATURES], base_train['y'])

# --- 3. Convergence Monitoring Loop ---
adaptation_steps = np.arange(1000, 21000, 1000)
results = {
    'TL_Train_Loss': [], 'TL_Test_Loss': [],
    'ML_Train_Loss': [], 'ML_Test_Loss': []
}

# Fixed 2018 Test Set for Testing Loss
test_df = df18.sample(2000, random_state=99)
pool_df = df18.drop(test_df.index)

for size in adaptation_steps:
    print(f"Adapting with {size} samples...")
    batch = pool_df.sample(n=size, random_state=42)
    
    # Adaptive TL Model
    tl_mod = copy.deepcopy(base_model)
    tl_mod.partial_fit(batch[TOP_22_FEATURES], batch['y'])
    results['TL_Train_Loss'].append(tl_mod.loss_)
    # Calculate Testing Loss manually using log_loss
    tl_probs = tl_mod.predict_proba(test_df[TOP_22_FEATURES])
    results['TL_Test_Loss'].append(log_loss(test_df['y'], tl_probs))
    
    # Traditional ML Model (From Scratch)
    ml_mod = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=42)
    ml_mod.partial_fit(batch[TOP_22_FEATURES], batch['y'], classes=[0, 1])
    results['ML_Train_Loss'].append(ml_mod.loss_)
    ml_probs = ml_mod.predict_proba(test_df[TOP_22_FEATURES])
    results['ML_Test_Loss'].append(log_loss(test_df['y'], ml_probs))

# --- 4. Plotting ---
plt.figure(figsize=(12, 7))

# Plot Adaptive TL Curves
plt.plot(adaptation_steps, results['TL_Train_Loss'], 'b-', label='TL Training Loss')
plt.plot(adaptation_steps, results['TL_Test_Loss'], 'b--', alpha=0.6, label='TL Testing Loss')

# Plot Traditional ML Curves
plt.plot(adaptation_steps, results['ML_Train_Loss'], 'r-', label='ML Training Loss')
plt.plot(adaptation_steps, results['ML_Test_Loss'], 'r--', alpha=0.6, label='ML Testing Loss')

plt.title('Training vs. Testing Loss Convergence (TL vs. Traditional ML)')
plt.xlabel('2018 Adaptation Samples')
plt.ylabel('Log-Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('train_test_loss_comparison.png')
plt.show()

# --- 5. Quantitative Generalization Analysis ---
final_idx = -1 # The 20,000 sample mark
tl_final_test_loss = results['TL_Test_Loss'][final_idx]
ml_final_test_loss = results['ML_Test_Loss'][final_idx]

# Calculate Generalization Ratio
gr = ml_final_test_loss / tl_final_test_loss

# Calculate Generalization Gap (Internal consistency)
tl_gap = results['TL_Test_Loss'][final_idx] - results['TL_Train_Loss'][final_idx]
ml_gap = results['ML_Test_Loss'][final_idx] - results['ML_Train_Loss'][final_idx]

print(f"\n--- Generalization Metrics at {adaptation_steps[final_idx]} samples ---")
print(f"Adaptive TL Test Loss: {tl_final_test_loss:.4f}")
print(f"Traditional ML Test Loss: {ml_final_test_loss:.4f}")
print(f"Generalization Ratio (GR): {gr:.2f}")
print(f"TL Internal Gap: {tl_gap:.4f}")
print(f"ML Internal Gap: {ml_gap:.4f}")
