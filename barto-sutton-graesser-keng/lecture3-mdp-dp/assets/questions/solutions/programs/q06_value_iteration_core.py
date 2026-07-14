"""
Value Iteration Core: Single line of code for the V[s] update in Value Iteration.
"""
import numpy as np

def value_iteration_update(s, actions, dynamics, V, gamma):
    # V[s] = max(sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in dynamics[(s, a)]) for a in actions)
    return max(sum(p * (r + gamma * V[s_prime]) 
                   for p, s_prime, r in dynamics[(s, a)]) 
               for a in actions)

if __name__ == "__main__":
    V = {0: 0.0, 1: 0.0}
    actions = ["a1", "a2"]
    dynamics = {
        (0, "a1"): [(1.0, 1, 5.0)],
        (0, "a2"): [(1.0, 0, 2.0)]
    }
    new_v = value_iteration_update(0, actions, dynamics, V, 0.9)
    print(f"Updated V[0] after one value iteration step: {new_v}")
