import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
from scipy.integrate import quad
from tqdm import tqdm

# ========== 1. Rejection Sampling ==========
def h(x):
    """Unnormalized target function"""
    return x**2 * (np.sin(np.pi*x))**2

# Rejection sampling parameters
x_min, x_max = -3, 3
c = 6.25  # Upper bound of h(x) in [-3,3]

def rejection_sampling(n_samples):
    """Rejection sampling without knowing A"""
    samples = []
    while len(samples) < n_samples:
        x = uniform.rvs(loc=x_min, scale=x_max-x_min)  # g(x) = uniform
        u = uniform.rvs()  # Uniform [0,1]
        if u <= h(x) / c:  # Acceptance probability
            samples.append(x)
    return np.array(samples)

# Generate 20,000 samples
np.random.seed(42)
samples = rejection_sampling(20000)

# ========== 2. PDF Comparison ==========
A = 8.8480182  # Normalization constant
x_vals = np.linspace(x_min, x_max, 1000)
pdf_theory = h(x_vals) / A

plt.figure(figsize=(10, 5))
plt.hist(samples, bins=50, density=True, alpha=0.6,
         color='skyblue', edgecolor='white',
         label='Empirical PDF (n=20,000)')
plt.plot(x_vals, pdf_theory, 'r-', lw=2,
         label='Theoretical PDF')
plt.title("Empirical vs Theoretical PDF Comparison")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ========== 3. Confidence Intervals ==========
data = samples[:200]
n = len(data)

# Classical CI formulas
def classic_ci(data, stat, alpha=0.05):
    """Classical confidence intervals"""
    if stat == 'mean':
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        z = norm.ppf(1 - alpha/2)
        return (mean - z*se, mean + z*se)
    elif stat == 'median':
        from scipy.stats import iqr
        f_hat = 1 / (iqr(data) / 1.349)  # Density estimator
        se = 1 / (2 * f_hat * np.sqrt(n))
        median = np.median(data)
        return (median - 1.96*se, median + 1.96*se)
    elif stat == 'quantile':
        p = 0.9
        quantile = np.quantile(data, p)
        f_quantile = h(quantile) / A
        se = np.sqrt(p*(1-p)/n) / f_quantile
        return (quantile - 1.96*se, quantile + 1.96*se)

# Bootstrap CI
def bootstrap_ci(data, stat, alpha=0.05, B=9999):
    """Bootstrap confidence intervals"""
    stats = []
    for _ in range(B):
        resample = np.random.choice(data, size=n, replace=True)
        if stat == 'mean':
            stats.append(np.mean(resample))
        elif stat == 'median':
            stats.append(np.median(resample))
        elif stat == 'quantile':
            stats.append(np.quantile(resample, 0.9))
    return np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])

# Compute and print results
print("\n=== Confidence Interval Results ===")
for stat in ['mean', 'median', 'quantile']:
    classic = classic_ci(data, stat)
    boot = bootstrap_ci(data, stat)
    name = '0.9-Quantile' if stat == 'quantile' else stat.capitalize()
    print(f"{name} - Classical: [{classic[0]:.4f}, {classic[1]:.4f}]")
    print(f"{name} - Bootstrap: [{boot[0]:.4f}, {boot[1]:.4f}]\n")

# ========== 4. Coverage Validation ==========
# Calculate true mean
true_mean, _ = quad(lambda x: x * h(x) / A, x_min, x_max)
print(f"Theoretical Mean: {true_mean:.10f}")

# Split into 100 subsets
subsets = samples.reshape(100, 200)
contain_count = 0

for subset in tqdm(subsets, desc="Validating coverage"):
    mean = np.mean(subset)
    se = np.std(subset, ddof=1) / np.sqrt(200)
    z = norm.ppf(0.975)
    ci_low = mean - z * se
    ci_high = mean + z * se
    if ci_low <= true_mean <= ci_high:
        contain_count += 1

print("\n=== Coverage Results ===")
print(f"Expected coverage: 95%")
print(f"Actual coverage: {contain_count}% ({contain_count}/100)")