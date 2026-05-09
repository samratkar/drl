"""
Efficiency in Summation: Optimize summation for sparse transitions.
"""
import numpy as np

def efficient_sum(s, a, sparse_dynamics, V, gamma):
    # sparse_dynamics[(s, a)] only contains entries where p > 0
    return sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in sparse_dynamics[(s, a)])

if __name__ == "__main__":
    # Sparse dynamics: only non-zero probability transitions are stored
    sparse_dynamics = {
        (0, "a"): [(1.0, 1, 5.0)] 
    }
    V = {1: 10.0}
    val = efficient_sum(0, "a", sparse_dynamics, V, 0.9)
    print(f"Value calculated from sparse transitions: {val}")
