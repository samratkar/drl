"""
Model-Based vs. Model-Free: Explain/show difference in input types.
"""
import numpy as np

def model_based_update(s, a, model, V, gamma):
    # 'model' is p(s', r | s, a)
    # Model-Based: Iterates over ALL possible outcomes.
    return sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in model[(s, a)])

def model_free_update(s, a, r, s_prime, V, gamma, alpha):
    # No model, just the transition experience (sample)
    # Model-Free: Updates using a single sample and a learning rate.
    target = r + gamma * V[s_prime]
    V[s] += alpha * (target - V[s])

if __name__ == "__main__":
    # Model-based example
    V_mb = {0: 0.0, 1: 10.0}
    model = {(0, "go"): [(0.5, 0, 0), (0.5, 1, 10)]}
    new_v = model_based_update(0, "go", model, V_mb, 0.9)
    print(f"Model-based updated value: {new_v}")
    
    # Model-free example
    V_mf = {0: 0.0, 1: 10.0}
    model_free_update(0, "go", 10, 1, V_mf, 0.9, 0.1)
    print(f"Model-free updated value: {V_mf[0]}")
