import numpy as np

def exercise1():
    n = 10_000_000

def exercise2():
    # Setting sample size
    n = 10_000_000

    # Generate a sample exponential distribution（μ=1，scale=1）
    exp_samples = np.random.exponential(scale=1, size=n)

    # Generate uniformly distributed samples (interval [0,5])
    uniform_samples = np.random.uniform(low=0, high=5, size=n)

    # Calculate the probability that the exponential variable is larger than the uniform variable
    prob_simulation = np.mean(exp_samples > uniform_samples)

    # Theoretical probability
    prob_theoretical = (1 - np.exp(-5)) / 5

    print(f"Simulated probability: {prob_simulation:.4f}")
    print(f"Theoretical probability: {prob_theoretical:.4f}")


if __name__ == "__main__":
    exercise1()
    exercise2()