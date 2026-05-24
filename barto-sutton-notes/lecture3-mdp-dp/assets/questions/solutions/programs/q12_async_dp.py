"""
Asynchronous DP (Random): Snippet to update a random state.
"""
import numpy as np

def async_update_random_state(states, V, actions, dynamics, gamma):
    s = np.random.choice(states)
    # Perform a single value iteration update on state s
    V[s] = max(sum(p * (r + gamma * V[s_prime]) 
                   for p, s_prime, r in dynamics[(s, a)]) 
               for a in actions)
    return s

if __name__ == "__main__":
    states = [0, 1, 2]
    V = {s: 0.0 for s in states}
    actions = ["a"]
    dynamics = { (s, "a"): [(1.0, (s+1)%3, 1.0)] for s in states }
    updated_s = async_update_random_state(states, V, actions, dynamics, 0.9)
    print(f"Randomly picked state {updated_s} and updated its value.")
    print(f"Current V: {V}")
