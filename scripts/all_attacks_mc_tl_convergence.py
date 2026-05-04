import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import copy

def generate_domain_data(n_samples, attack_type, domain_shift=0.5, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 14) + domain_shift
    
    if attack_type == 'DDoS':
        logits = 1.2 * X[:, 4] + 0.8 * X[:, 3]
    elif attack_type == 'Brute Force':
        logits = 1.0 * X[:, 2] + 0.6 * X[:, 5]
    elif attack_type == 'Infiltration':
        logits = 0.5 * X[:, 0] + 0.4 * X[:, 1]
    elif attack_type == 'Botnet':
        logits = 0.9 * X[:, 6] + 0.7 * X[:, 7]
    elif attack_type == 'Web Attack':
        logits = 0.8 * X[:, 8] + 0.6 * X[:, 9]
    else:
        logits = np.random.normal(-2, 0.5, n_samples)
        
    y = (logits > 0.5).astype(int)
    return X, y

attack_categories = ['DDoS', 'Brute Force', 'Infiltration', 'Botnet', 'Web Attack']
test_sizes = [100, 500, 1000, 2500, 5000, 7200, 10000]
n_iterations = 10
mc_all_results = {}

for attack in attack_categories:
    # Pre-train base model (Source 2017)
    X_src, y_src = generate_domain_data(5000, attack, domain_shift=0.0, seed=1)
    base_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    base_model.fit(X_src, y_src)
    
    # Target Domain Test Set
    X_test, y_test = generate_domain_data(2000, attack, domain_shift=0.5, seed=99)
    
    attack_mc_matrix = np.zeros((n_iterations, len(test_sizes)))
    
    for i in range(n_iterations):
        for j, size in enumerate(test_sizes):
            # Fine-tune on adaptation subset with iteration-specific seed
            X_adapt, y_adapt = generate_domain_data(size, attack, domain_shift=0.5, seed=size+i*100)
            tl_model = copy.deepcopy(base_model)
            tl_model.partial_fit(X_adapt, y_adapt)
            attack_mc_matrix[i, j] = f1_score(y_test, tl_model.predict(X_test))
            
    mc_all_results[attack] = np.mean(attack_mc_matrix, axis=0)

# Create DataFrame for plotting
df_mc = pd.DataFrame(mc_all_results, index=test_sizes)

# Plotting
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'purple', 'red']
for idx, attack in enumerate(attack_categories):
    plt.plot(test_sizes, df_mc[attack], marker='o', label=f"{attack} (MC Avg)", color=colors[idx])

plt.axvline(x=7200, color='black', linestyle='--', label='Reliability Threshold')
plt.title('Monte Carlo Transfer Learning Convergence: All Attack Types')
plt.xlabel('Adaptation Samples (Target Domain)')
plt.ylabel('Averaged F1 Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('all_attacks_mc_tl_convergence.png')

df_mc.to_csv('mc_tl_experiment_results.csv')
print(df_mc)