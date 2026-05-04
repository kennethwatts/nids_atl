import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. SETUP & DATA LOADING
def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 2. CALCULATION
window_size = 50
quantile_val = 0.95
num_samples = 1500 

X18 = scaler.transform(df18[features][:num_samples]) 
y18 = (df18['Label'][:num_samples] != 'benign').astype(int).values
probs = expert.predict_proba(X18)[:, 1]

thresh_lagged = [np.nan] * window_size 
thresh_current = []
current_lag_val = np.nan 

for i in range(len(probs)):
    window_start = (i // window_size) * window_size
    current_window_data = probs[window_start : window_start + window_size]
    thresh_current.append(np.quantile(current_window_data, quantile_val))
    
    if i >= window_size:
        if i % window_size == 0:
            current_lag_val = np.quantile(probs[i-window_size : i], quantile_val)
        thresh_lagged.append(current_lag_val)

# 3. SEGMENT ANALYSIS (Samples 100 to 300)
seg_start, seg_end = 100, 300
s_probs = probs[seg_start:seg_end]
s_labels = y18[seg_start:seg_end]
s_t_curr = np.array(thresh_current[seg_start:seg_end])
s_t_lag = np.array(thresh_lagged[seg_start:seg_end])

alerts_curr = (s_probs >= s_t_curr).astype(int)
alerts_lag = (s_probs >= s_t_lag).astype(int)

# 4. GENERATE TABLE
comparison_df = pd.DataFrame({
    'Metric': [
        'Total Flow Samples Analyzed',
        'Shared Alerts (Both Agreed)',
        'Extra Alerts (Lagged Only)',
        'Missed Alerts (Lagged vs. Oracle)',
        'True Positives (Oracle Strategy)',
        'True Positives (Lagged Strategy)'
    ],
    'Value': [
        seg_end - seg_start,
        np.sum((alerts_curr == 1) & (alerts_lag == 1)),
        np.sum((alerts_lag == 1) & (alerts_curr == 0)),
        np.sum((alerts_lag == 0) & (alerts_curr == 1)),
        np.sum((alerts_curr == 1) & (s_labels == 1)),
        np.sum((alerts_lag == 1) & (s_labels == 1))
    ]
})

comparison_df.to_csv("alert_strategy_comparison_table.csv", index=False)
print(comparison_df)