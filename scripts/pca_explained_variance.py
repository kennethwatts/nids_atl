import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. LOAD AND SANITIZE (Using 2018 as the target for "Ideal" features)
file_path = "cic2018_training_av_lbl_100K_hdr.csv"
df = pd.read_csv(file_path)

# Drop labels and handle infinites/NaNs
X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)

# 2. SCALE THE DATA (Crucial for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. RUN PCA
pca = PCA()
pca.fit(X_scaled)

# Calculate Cumulative Explained Variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 4. FIND ELBOW / OPTIMAL COMPONENTS (95% Variance Threshold)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

# 5. IDENTIFY TOP CONTRIBUTING FEATURES
# We look at the "loadings" of the first few principal components
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(len(X.columns))], 
    index=X.columns
)

# Sum absolute loadings for the top components that explain 95% variance
top_features = loadings.iloc[:, :n_components_95].abs().sum(axis=1).sort_values(ascending=False)

# 6. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='#2980b9')
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')
plt.axvline(x=n_components_95, color='g', linestyle=':', label=f'Optimal Components: {n_components_95}')

plt.title('PCA Explained Variance: Determining Feature Dimensionality', fontsize=14)
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_explained_variance.png')
plt.show()

print(f"Optimal Number of Components (95% Variance): {n_components_95}")
print("\n--- TOP 10 IDEAL FEATURES (Based on PCA Loadings) ---")
print(top_features.head(10))