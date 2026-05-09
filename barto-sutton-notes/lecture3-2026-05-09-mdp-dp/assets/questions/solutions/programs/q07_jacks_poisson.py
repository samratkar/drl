"""
Jack's Car Rental (Poisson): Pre-compute Poisson probabilities up to max_cars.
"""
import numpy as np
from scipy.stats import poisson

def precompute_poisson(max_cars, lam):
    # Returns a list where index k is P(X=k)
    return [poisson.pmf(k, lam) for k in range(max_cars + 1)]

if __name__ == "__main__":
    probs = precompute_poisson(5, 3)
    print("Poisson probabilities for lambda=3 up to 5 cars:")
    for k, p in enumerate(probs):
        print(f"P(X={k}) = {p:.4f}")
