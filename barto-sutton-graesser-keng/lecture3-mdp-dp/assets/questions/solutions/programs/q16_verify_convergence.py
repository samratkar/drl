"""
Verifying Convergence: Function is_optimal(V, P, gamma, theta).
"""
import numpy as np

def is_optimal(V, states, actions, P, gamma, theta):
    """
    Checks if V satisfies the Bellman Optimality Equation within tolerance theta.
    """
    for s in states:
        max_q = max(sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in P[(s, a)]) 
                    for a in actions)
        if abs(V[s] - max_q) > theta:
            return False
    return True

if __name__ == "__main__":
    V = {0: 10.0}
    states = [0]
    actions = ["a"]
    P = {(0, "a"): [(1.0, 0, 1.0)]}
    # For V=10, max_q = 1 + 0.9*10 = 10. Satisfies optimality.
    print(f"Is V={V} optimal? {is_optimal(V, states, actions, P, 0.9, 0.01)}")
