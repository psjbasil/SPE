import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------ 1. Load data and plot scatter ------------------
data = pd.read_csv("data_ex1_wt.csv", header=None)
time = data[0].values
measurements = data[1].values

plt.figure(figsize=(10, 5))
plt.scatter(time, measurements, s=1)
plt.title("Scatter Plot of Measurements over Time")
plt.xlabel("Time (s)")
plt.ylabel("Measurement")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ 2. Least squares polynomial trend (degree 5) ------------------
degree = 5
coeffs = np.polyfit(time, measurements, deg=degree)
trend = np.polyval(coeffs, time)
detrended = measurements - trend

# ------------------ 3. Plot histogram of detrended data ------------------
plt.figure(figsize=(10, 5))
plt.hist(detrended, bins=50, density=True, alpha=0.7, color='skyblue')
plt.title("Histogram of De-trended Data")
plt.xlabel("Residual")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ 4. EM Algorithm for Gaussian Mixture Model ------------------
def em_gmm(data, n_components=3, max_iter=100, tol=1e-6):
    np.random.seed(0)
    n = len(data)
    data = data.reshape(-1, 1)

    # Initialization
    means = np.random.choice(data.flatten(), n_components)
    variances = np.full(n_components, np.var(data))
    weights = np.full(n_components, 1 / n_components)

    log_likelihoods = []

    for iteration in range(max_iter):
        # E-step
        responsibilities = np.zeros((n, n_components))
        for k in range(n_components):
            responsibilities[:, k] = weights[k] * norm.pdf(data.flatten(), means[k], np.sqrt(variances[k]))
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step
        N_k = responsibilities.sum(axis=0)
        means = (responsibilities * data).sum(axis=0) / N_k
        variances = ((responsibilities * (data - means)**2).sum(axis=0)) / N_k
        weights = N_k / n

        # Log-likelihood
        ll = np.sum(np.log(np.sum([
            weights[k] * norm.pdf(data.flatten(), means[k], np.sqrt(variances[k]))
            for k in range(n_components)
        ], axis=0)))
        log_likelihoods.append(ll)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return means, variances, weights, log_likelihoods[-1]

# Fit and plot for 3 Gaussians
means, variances, weights, final_ll = em_gmm(detrended, n_components=3)

# Plot histogram and fitted PDFs
x_vals = np.linspace(min(detrended), max(detrended), 1000)
plt.figure(figsize=(10, 5))
plt.hist(detrended, bins=50, density=True, alpha=0.6, label='Empirical Histogram')

pdf_total = np.zeros_like(x_vals)

sorted_indices = np.argsort(means)
sorted_means = means[sorted_indices]
sorted_variances = variances[sorted_indices]
sorted_weights = weights[sorted_indices]

for i in range(3):
    pdf = sorted_weights[i] * norm.pdf(x_vals, sorted_means[i], np.sqrt(sorted_variances[i]))
    pdf_total += pdf
    plt.plot(x_vals, pdf, label=f'Gaussian {i+1} (μ={sorted_means[i]:.2f}, σ²={sorted_variances[i]:.2f})')

plt.plot(x_vals, pdf_total, 'k--', label='Mixture PDF')
plt.title("Fitted Gaussian Mixture Model (3 Components)")
plt.xlabel("Residual")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print estimated parameters
print("=== GMM Parameters ===")
for i in range(3):
    print(f"Component {i+1}: μ = {sorted_means[i]:.3f}, σ² = {sorted_variances[i]:.3f}, π = {sorted_weights[i]:.3f}")

# ------------------ Compare with true parameters ------------------
true_means = np.array([-5, 0, 4])
true_variances = np.array([3, 6, 1])

sorted_indices = np.argsort(means)
sorted_means = means[sorted_indices]
sorted_variances = variances[sorted_indices]
sorted_weights = weights[sorted_indices]

print("=== Comparison with Given Parameters ===")
print(f"{'Component':<10}{'True μ':>10}{'Est. μ':>10}{'Δμ':>10}{'True σ²':>12}{'Est. σ²':>12}{'Δσ²':>10}")
for i in range(3):
    mu_diff = sorted_means[i] - true_means[i]
    var_diff = sorted_variances[i] - true_variances[i]
    print(f"{i+1:<10}{true_means[i]:>10.2f}{sorted_means[i]:>10.2f}{mu_diff:>10.2f}"
          f"{true_variances[i]:>12.2f}{sorted_variances[i]:>12.2f}{var_diff:>10.2f}")

# ------------------ 5. Optional: Determine optimal number of Gaussians (BIC) ------------------
def compute_bic(log_likelihood, n_params, n_samples):
    return n_params * np.log(n_samples) - 2 * log_likelihood

bic_scores = []
component_range = range(1, 6)

for k in component_range:
    _, _, _, ll = em_gmm(detrended, n_components=k)
    n_params = k - 1 + k * 2  # weights + means + variances
    bic = compute_bic(ll, n_params, len(detrended))
    bic_scores.append(bic)
    print(f"Components = {k}: BIC = {bic:.2f}")

plt.figure(figsize=(8, 4))
plt.plot(component_range, bic_scores, marker='o')
plt.title("BIC for Different Number of Gaussian Components")
plt.xlabel("Number of Components")
plt.ylabel("BIC")
plt.grid(True)
plt.tight_layout()
plt.show()

best_k = component_range[np.argmin(bic_scores)]
print(f"Optimal number of components (by BIC): {best_k}")