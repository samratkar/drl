"""
Policy Improvement Logic: Logic to find the greedy action for state s.
"""
import numpy as np

def get_greedy_action(s, V, actions, dynamics, gamma):
    action_values = []
    for a in actions:
        q_sa = sum(p * (r + gamma * V[s_prime]) 
                   for p, s_prime, r in dynamics[(s, a)])
        action_values.append(q_sa)
    return actions[np.argmax(action_values)]

if __name__ == "__main__":
    V = {0: 1.0, 1: 2.0}
    actions = ["search", "wait"]
    dynamics = {
        (0, "search"): [(1.0, 0, 1.0)],
        (0, "wait"): [(1.0, 1, 0.5)]
    }
    greedy_action = get_greedy_action(0, V, actions, dynamics, 0.9)
    print(f"Greedy action for state 0: {greedy_action}")
