"""
Policy Stability Check: Function are_policies_equal(pi1, pi2).
"""
import numpy as np

def are_policies_equal(pi1, pi2, states):
    """
    Returns True if policies provide the same action for all states.
    """
    return all(pi1[s] == pi2[s] for s in states)

if __name__ == "__main__":
    states = [0, 1, 2]
    pi1 = {0: "a", 1: "b", 2: "c"}
    pi2 = {0: "a", 1: "b", 2: "c"}
    pi3 = {0: "a", 1: "b", 2: "a"}
    
    print(f"pi1 == pi2? {are_policies_equal(pi1, pi2, states)}")
    print(f"pi1 == pi3? {are_policies_equal(pi1, pi3, states)}")
