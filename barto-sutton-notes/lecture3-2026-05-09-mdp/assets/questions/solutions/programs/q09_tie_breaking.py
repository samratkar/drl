"""
Tie-breaking: argmax that handles multiple max actions.
"""
import numpy as np

def stable_argmax(values):
    """
    Returns a random choice among the indices that have the maximum value.
    """
    max_val = np.max(values)
    indices = np.where(np.array(values) == max_val)[0]
    return np.random.choice(indices)

if __name__ == "__main__":
    values = [10, 5, 10, 8]
    choice = stable_argmax(values)
    print(f"Values: {values}")
    print(f"Stable argmax choice (index): {choice}")
