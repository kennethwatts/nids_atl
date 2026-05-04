import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def generate_ids_variant_data(n_samples, attack_type, seed):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 14)
    
    # Behavioral logic for a wider range of attacks
    if attack_type == 'DDoS':
        logits = 1.2 * X[:, 4] + 0.8 * X[:, 3]
    elif attack_type == 'Brute Force':
        logits = 1.0 * X[:, 2] + 0.6 * X[:, 5]
    elif attack_type == 'Infiltration':
        logits = 0.5 * X[:, 0] + 0.4 * X[:, 1]
    elif attack_type == 'Botnet':
        # C2 communication: periodicity in inter-arrival times
        logits = 0.9 * X[:, 6] + 0.7 * X[:, 7]
    elif attack_type == 'Web Attack':
        # Cross-site scripting / SQLi: specific packet length variance
        logits = 0.8 * X[:, 8] + 0.6 * X[:, 9]
    else: # Benign
        logits = np.random.normal(-2, 0.5, n_samples)
        
    y = (logits > 0).astype(int)
    return X, y

# Expanded categories based on CSE-CIC-IDS2018 classes
attack_categories = ['DDoS', 'Brute Force', 'Infiltration', 'Botnet', 'Web Attack']
test_sizes = [100, 500, 1000, 2500, 5000, 7200, 10000]
f1_results = {}

for attack in attack_categories:
    X_test, y_test = generate_ids_variant_data(2000, attack, seed=99)
    attack_f1s = []
    
    for size in test_sizes:
        X_train, y_train = generate_ids_variant_data(size, attack, seed=size)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        score = f1_score(y_test, y_pred)
        attack_f1s.append(score)
    
    f1_results[attack] = attack_f1s

# Plotting F1 Comparison
plt.figure(figsize=(10, 6))
for attack, f1s in f1_results.items():
    plt.plot(test_sizes, f1s, marker='o', label=attack)

plt.axvline(x=7200, color='red', linestyle='--', label='Reliability Gap')
plt.title('F1 Score Convergence Across All Primary Attack Types')
plt.xlabel('Training Samples')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('all_attack_f1_convergence.png')

print(pd.DataFrame(f1_results, index=test_sizes))