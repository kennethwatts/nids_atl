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

# 2. CALCULATION OVER 1,600 SAMPLES
window_size = 50
quantile_val = 0.95
num_samples = 1600 # Expanded to match your graph

X18 = scaler.transform(df18[features][:num_samples]) 
y18 = (df18['Label'][:num_samples] != 'benign').astype(int).values
probs = expert.predict_proba(X18)[:, 1]

thresh_lagged = [np.nan] * window_size 
thresh_current = []
current_lag_val = np.nan 

for i in range(len(probs)):
    # Oracle Strategy (Current Window)
    window_start = (i // window_size) * window_size
    current_window_data = probs[window_start : window_start + window_size]
    thresh_current.append(np.quantile(current_window_data, quantile_val))
    
    # Practical Strategy (Lagged Window)
    if i >= window_size:
        if i % window_size == 0:
            current_lag_val = np.quantile(probs[i-window_size : i], quantile_val)
        thresh_lagged.append(current_lag_val)

# 3. COMPREHENSIVE ANALYSIS (Post-Initialization)
# We exclude the first 50 samples where the lagged threshold wasn't available
analysis_start = window_size 
a_probs = probs[analysis_start:num_samples]
a_labels = y18[analysis_start:num_samples]
a_t_curr = np.array(thresh_current[analysis_start:num_samples])
a_t_lag = np.array(thresh_lagged[analysis_start:num_samples])

alerts_curr = (a_probs >= a_t_curr).astype(int)
alerts_lag = (a_probs >= a_t_lag).astype(int)

# 4. GENERATE FINAL TABLE
final_comparison_df = pd.DataFrame({
    'Metric': [
        'Total Samples Analyzed (excluding warm-up)',
        'Shared Alerts (Both Methods Agreed)',
        'Extra Alerts (Lagged Strategy Only)',
        'Missed Alerts (Lagged vs. Oracle Strategy)',
        'True Positives (Oracle / Theoretical)',
        'True Positives (Lagged / Practical)'
    ],
    'Value': [
        len(a_probs),
        np.sum((alerts_curr == 1) & (alerts_lag == 1)),
        np.sum((alerts_lag == 1) & (alerts_curr == 0)),
        np.sum((alerts_lag == 0) & (alerts_curr == 1)),
        np.sum((alerts_curr == 1) & (a_labels == 1)),
        np.sum((alerts_lag == 1) & (a_labels == 1))
    ]
})

final_comparison_df.to_csv("alert_strategy_comparison_1600.csv", index=False)
print(final_comparison_df)