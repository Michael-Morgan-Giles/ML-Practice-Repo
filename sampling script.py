import math
from scipy.stats import norm

def calculate_sample_size_from_confusion_matrix(confidence_level, margin_of_error, confusion_matrix, ratio):
    """
    """
    TP = confusion_matrix['TP']
    TN = confusion_matrix['TN']
    FP = confusion_matrix['FP']
    FN = confusion_matrix['FN']
    
    
    total = TP + TN + FP + FN
    expected_proportion = (TP + FP) / total
    
    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - alpha / 2)
    
    total_sample_size = (z_score**2 * expected_proportion * (1 - expected_proportion)) / margin_of_error**2
    total_sample_size = math.ceil(total_sample_size)
    
    control_sample_size = math.ceil(total_sample_size / (1 + ratio))
    treated_sample_size = total_sample_size - control_sample_size
    
    return control_sample_size, treated_sample_size

# Example usage
confidence_level = 0.95
margin_of_error = 0.05
confusion_matrix = {'TP': 50, 'TN': 30, 'FP': 10, 'FN': 10}
confusion_matrix = {'TP': 16000, 'TN': 500000, 'FP': 4000, 'FN': 600000}
ratio = 3  

control_size, treated_size = calculate_sample_size_from_confusion_matrix(confidence_level, margin_of_error, confusion_matrix, ratio)
print(f"The required sample size is: {control_size} for control and {treated_size} for treated.")

z_score = norm.ppf(1 - 0.05 / 2)
math.ceil((z_score**2 * 0.95 * (1 - 0.05)) / margin_of_error**2)

#%%
import math
from scipy.stats import norm
import numpy as np

#binomial dist sample calculation
def calculate_sample_size(p, confidence_level, margin_of_error):
    
    z = norm.ppf((1 + confidence_level) / 2)
    
    n = (z**2 * p * (1 - p)) / (margin_of_error**2)
    return math.ceil(n)


confidence_level = 0.95
margin_of_error = 0.05

# Example confusion matrix
confusion_matrix = np.array([
    [16000, 800000],  # True Positives, False Negatives
    [4000, 600000]    # False Positives, True Negatives
])

TP = confusion_matrix[0, 0]
FN = confusion_matrix[0, 1]
FP = confusion_matrix[1, 0]
TN = confusion_matrix[1, 1]

# PPV & NPV  & PP
PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
PP = (TP + FP) / confusion_matrix.sum()

sample_size_ppv = calculate_sample_size(PPV, confidence_level, margin_of_error)
sample_size_npv = calculate_sample_size(NPV, confidence_level, margin_of_error)
sample_size_pp = calculate_sample_size(PP, confidence_level, margin_of_error)


total_sample_size = max(sample_size_ppv, sample_size_npv)


treated_to_control_ratio = 3
control_size = total_sample_size / (treated_to_control_ratio + 1)
treated_size = treated_to_control_ratio * control_size

print(f"Sample size for PPV: {sample_size_ppv}")
print(f"Sample size for NPV: {sample_size_npv}")
print(f"Total sample size (larger of PPV/NPV): {total_sample_size}")
print(f"Control group size: {control_size}")
print(f"Treated group size: {treated_size}")


# combinations of all variables as a loop or in a dataframe
prob = np.arange(0.01,0.5,0.01)
me_list = np.arange(0.01,0.1,0.01)
CI = np.arange(0.9,1,0.01)

for p in prob:
    for m in me_list:
        for c in CI:
            results = []
            results.append(calculate_sample_size(p, c, m))
            print(results)

from itertools import product
import pandas as pd

# combinations
combinations = list(product(prob, me_list, CI))

df = pd.DataFrame(combinations, columns=['prob', 'me_list', 'CI'])

df['control_sample_size'] = ((norm.ppf((1 + df['CI']) / 2))**2 * df['prob'] * (1 - df['prob'])) / (df['me_list']**2)

print(df)



#%% simple graphing example
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10000

control_group = np.random.normal(loc=50, scale=10, size=n_samples)
treated_group = np.random.normal(loc=100, scale=10, size=n_samples)


data = pd.DataFrame({
    'outcome': np.concatenate([control_group, treated_group]),
    'group': ['control'] * n_samples + ['treated'] * n_samples
})

plt.figure(figsize=(14, 6))

# Hist
plt.subplot(1, 2, 1)
sns.histplot(data, x='outcome', hue='group', element='step', stat='density', common_norm=False, kde=False)
plt.title('Histogram of Outcome Variable')
plt.xlabel('Outcome')
plt.ylabel('Density')

# KDE
plt.subplot(1, 2, 2)
sns.kdeplot(data=data, x='outcome', hue='group', common_norm=False, fill=True)
plt.title('Kernel Density Estimate of Outcome Variable')
plt.xlabel('Outcome')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
