"""
Gambler's Problem Stakes: Function get_possible_stakes(s, goal).
"""
import numpy as np

def get_possible_stakes(s, goal):
    # Stake must be at least 0 and at most the amount you have (s) or need (goal - s)
    return range(min(s, goal - s) + 1)

if __name__ == "__main__":
    s = 40
    goal = 100
    stakes = list(get_possible_stakes(s, goal))
    print(f"Possible stakes for state {s} with goal {goal}: {stakes}")
