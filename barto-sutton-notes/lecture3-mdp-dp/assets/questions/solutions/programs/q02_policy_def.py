"""
Policy Definition: Function get_action(policy, state) for deterministic and stochastic policies.
"""
import numpy as np

def get_action(policy, state):
    """
    Handles both deterministic: {s: a} and stochastic: {s: {a: prob}} policies.
    """
    action_info = policy[state]
    if isinstance(action_info, dict):
        # Stochastic policy
        actions = list(action_info.keys())
        probs = list(action_info.values())
        return np.random.choice(actions, p=probs)
    # Deterministic policy
    return action_info

if __name__ == "__main__":
    det_policy = {0: "search", 1: "wait"}
    stoch_policy = {0: {"search": 0.7, "wait": 0.3}, 1: {"wait": 1.0}}
    
    print(f"Deterministic action for state 0: {get_action(det_policy, 0)}")
    print(f"Stochastic action for state 0: {get_action(stoch_policy, 0)}")
