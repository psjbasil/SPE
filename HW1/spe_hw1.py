import numpy as np

def exercise1():
    # Set sample size
    n = 1_000_000

    # Set Parameters
    mus = np.array([-2, 4, 10, 15])
    vars_ = np.array([2, 1, 3, 2])
    probs = np.array([0.15, 0.25, 0.35, 0.25])

    # Generate distribution indices according to probabilities
    Z = np.random.choice(4, size=n, p=probs)

    # Generate data from corresponding normal distributions based on Z
    X = np.random.normal(loc=mus[Z], scale=np.sqrt(vars_[Z]))

    # Calculate overall expectation and variance
    mean_X = np.mean(X)
    var_X = np.var(X)

    # Calculate theoretical expectation and variance decomposition
    E_X_given_Z = np.sum(probs * mus)
    Var_X_given_Z = np.sum(probs * vars_)
    Var_E_X_given_Z = np.sum(probs * (mus - E_X_given_Z)**2)

    # Print verification results
    print(f"Excercise 1 Empirical E[X]: {mean_X}, Theoretical E[E[X|Z]]: {E_X_given_Z}")
    # Excercise 1 Empirical E[X]: 7.94946969949475, Theoretical E[E[X|Z]]: 7.95
    print(f"Excercise 1 Empirical Var(X): {var_X}, Theoretical Var: {Var_X_given_Z + Var_E_X_given_Z}")
    # Excercise 1 Empirical Var(X): 34.75545902873244, Theoretical Var: 34.747499999999995

def exercise2():
    # Setting sample size
    n = 10_000_000

    # Generate a sample exponential distribution（μ=1，scale=1）
    exp_samples = np.random.exponential(scale=1, size=n)

    # Generate uniformly distributed samples (interval [0,5])
    uniform_samples = np.random.uniform(low=0, high=5, size=n)

    # Calculate the probability that the exponential variable is larger than the uniform variable
    prob_simulation = np.mean(exp_samples > uniform_samples)

    print(f"Exercise 2 Simulated probability: {prob_simulation:.4f}")



if __name__ == "__main__":
    exercise1()
    exercise2()