"""
Dynamics Representation: p(s', r | s, a) in a dictionary.
Representing as {(state, action): [(probability, next_state, reward), ...]}
"""
import numpy as np

dynamics = {
    (0, "search"): [(0.7, 0, 1), (0.3, 1, 1)],
    (0, "wait"): [(1.0, 0, 0.5)],
    (1, "search"): [(0.8, 1, 1), (0.2, 0, -3)],
    (1, "wait"): [(1.0, 1, 0.5)],
    (1, "recharge"): [(1.0, 0, 0)]
}

if __name__ == "__main__":
    print("Dynamics for state 0, action 'search':")
    print(dynamics[(0, "search")])
