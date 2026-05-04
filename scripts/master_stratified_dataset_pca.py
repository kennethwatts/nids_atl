import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. LOAD AND CLEAN (Handling the UTF-8-SIG and mixed types)
file_path = "master_stratified_dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)

# Mapping to categories for a cleaner plot
category_map = {
    'DoS GoldenEye': 'DoS', 'DoS Hulk': 'DoS', 'DoS Slowhttptest': 'DoS', 'DoS slowloris': 'DoS',
    'DDoS': 'DDoS', 'SSH-Bruteforce': 'Brute Force', 'FTP-Patator': 'Brute Force',
    'Web Attack - Brute Force': 'Web', 'Web Attack - XSS': 'Web', 'Web Attack - Sql Injection': 'Web',
    'Bot': 'Botnet', 'Infiltration': 'Infiltration'
}
df['Category'] = df['Label'].astype(str).str.strip().map(category_map).fillna('Benign')

# Force Numeric and Clean
X = df.drop(columns=['Label', 'Category']).apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.median() if not np.isnan(x.median()) else 0), axis=0)

# 2. STANDARDIZATION (Critical for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA TRANSFORMATION
pca = PCA(n_components=0.95) # Capture 95% of variance
X_pca = pca.fit_transform(X_scaled)

# 4. VISUALIZATION: 2D CLUSTER MAP
plt.figure(figsize=(12, 8))
# We only plot a subset for clarity (e.g., 20k points) to avoid a messy overlap
plot_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
plot_df['Category'] = df['Category']
plot_df = plot_df.sample(n=min(len(plot_df), 20000), random_state=42)

sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Category', alpha=0.5, s=10)
plt.title("PCA Cluster Map: Attack Separation in Latent Space", fontsize=15)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_cluster_map.png')
plt.show()

# 5. CUMULATIVE VARIANCE (The "Elbow" Graph)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Variance Captured")
plt.grid(True)
plt.savefig('pca_variance_elbow.png')
plt.show()

print(f"Optimal Components for 95% Variance: {pca.n_components_}")