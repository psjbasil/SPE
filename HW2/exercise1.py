import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, uniform, kstest

# params
np.random.seed(42)  # ensure reproducibility
λ = 5
T = 1
N = 5  # λT ≈ N

# method 1: sample N arrival times uniformly
arrival_times_uniform = np.sort(np.random.uniform(0, T, N))
inter_arrival_times_uniform = np.diff(np.insert(arrival_times_uniform, 0, 0))

# method 2: generate N exponential inter-arrival times and accumulate to get arrival times
inter_arrival_times_exponential = np.random.exponential(1/λ, N)
arrival_times_exponential = np.cumsum(inter_arrival_times_exponential)

# only keep arrivals within [0, T]
arrival_times_exponential = arrival_times_exponential[arrival_times_exponential <= T]
N_exponential = len(arrival_times_exponential)

# analysis and Plotting

# method 1: check if inter-arrival times follow exponential distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(inter_arrival_times_uniform, bins=10, density=True, alpha=0.7, label="Simulated")
x = np.linspace(0, max(inter_arrival_times_uniform), 100)
plt.plot(x, λ*np.exp(-λ*x), 'r-', label="Theoretical Exponential PDF")
plt.title("Method 1: Inter-arrival times")
plt.legend()

# method 2: check if arrival times follow uniform distribution
plt.subplot(1, 2, 2)
plt.hist(arrival_times_exponential, bins=10, density=True, alpha=0.7, label="Simulated")
plt.plot(x, np.ones_like(x)/T, 'r-', label="Theoretical Uniform PDF")
plt.title("Method 2: Arrival times")
plt.legend()

plt.tight_layout()
plt.show()

# KS Test Quantification

# method 1 test: inter-arrival times ~ Exponential(λ)
D1, p_value1 = kstest(inter_arrival_times_uniform, 'expon', args=(0, 1/λ))

# method 2 test: arrival times ~ Uniform(0, T)
D2, p_value2 = kstest(arrival_times_exponential, 'uniform', args=(0, T))

print(f"method 1 - KS test p-value (should be large if exponential): {p_value1:.4f}")
print(f"method 2 - KS test p-value (should be large if uniform): {p_value2:.4f}")