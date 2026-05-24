"""
Absorbing State Implementation: Representing terminal states in dynamics.
A terminal state 'T' stays in 'T' with 0 reward for any action.
"""
import numpy as np

def add_absorbing_state(dynamics, terminal_state, actions):
    for a in actions:
        dynamics[(terminal_state, a)] = [(1.0, terminal_state, 0.0)]

if __name__ == "__main__":
    dynamics = {}
    actions = ["left", "right", "up", "down"]
    add_absorbing_state(dynamics, "goal", actions)
    print(f"Dynamics for absorbing state 'goal', action 'up': {dynamics[('goal', 'up')]}")
