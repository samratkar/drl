"""
In-place Update: Code for in-place update of V[s].
This updates the value function directly without keeping a copy of the old values.
"""
import numpy as np

def in_place_update(s, new_value, V):
    V[s] = new_value

if __name__ == "__main__":
    V = {0: 0.5, 1: 1.2}
    print(f"Initial V: {V}")
    in_place_update(0, 0.8, V)
    print(f"Updated V: {V}")
