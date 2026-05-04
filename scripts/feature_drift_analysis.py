import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Focus only on the 'math' features
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Label']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])

print("Analyzing Feature Drift...")
df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')

common_cols = list(set(df17.columns) & set(df18.columns))[:5] # Look at top 5

plt.figure(figsize=(12, 8))
for i, col in enumerate(common_cols):
    plt.subplot(2, 3, i+1)
    # Plot 2017 vs 2018 distribution
    sns.kdeplot(df17[col], label='2017', color='blue', shade=True)
    sns.kdeplot(df18[col], label='2018', color='red', shade=True)
    plt.title(f'Drift in: {col}')
    plt.legend()

plt.tight_layout()
plt.savefig('feature_drift_analysis.png')
plt.show()