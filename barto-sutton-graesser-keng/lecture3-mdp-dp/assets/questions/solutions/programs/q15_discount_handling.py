"""
Discount Factor Handling: calculate_q_value(s, a, V, P, gamma).
"""
import numpy as np

def calculate_q_value(s, a, V, P, gamma):
    """
    Calculates Q(s, a) using the transition model P and value function V.
    """
    return sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in P[(s, a)])

if __name__ == "__main__":
    P = {(0, "a"): [(0.8, 1, 10), (0.2, 0, 0)]}
    V = {0: 0, 1: 5}
    q_val = calculate_q_value(0, "a", V, P, 0.9)
    print(f"Calculated Q(0, 'a') with gamma=0.9: {q_val}")
