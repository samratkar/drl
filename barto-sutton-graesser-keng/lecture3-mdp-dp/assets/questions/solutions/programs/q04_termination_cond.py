"""
Termination Condition: while loop for Policy Evaluation with theta.
"""
import numpy as np

def policy_evaluation(policy, dynamics, states, gamma, theta):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # simplified bellman update for a specific policy
            # policy is assumed to be deterministic {s: a}
            new_v = sum(p * (r + gamma * V[s_prime]) 
                        for p, s_prime, r in dynamics[(s, policy[s])])
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
    return V

if __name__ == "__main__":
    states = [0, 1]
    policy = {0: "search", 1: "wait"}
    dynamics = {
        (0, "search"): [(0.7, 0, 1), (0.3, 1, 1)],
        (1, "wait"): [(1.0, 1, 0.5)]
    }
    V = policy_evaluation(policy, dynamics, states, 0.9, 0.01)
    print(f"Converged Value Function: {V}")
