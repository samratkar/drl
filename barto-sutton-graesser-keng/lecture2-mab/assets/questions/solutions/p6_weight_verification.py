"""P6. Exponential Recency Weights — Verification"""
import numpy as np

if __name__ == "__main__":
    alpha = 0.2
    Q1 = 0.0
    rewards = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    n = len(rewards)

    # Method 1: Incremental update
    Q = Q1
    for r in rewards:
        Q += alpha * (r - Q)
    Q_incremental = Q

    # Method 2: Closed-form
    Q_closed = (1 - alpha)**n * Q1
    for i, r in enumerate(rewards):
        Q_closed += alpha * (1 - alpha)**(n - 1 - i) * r

    print(f"Incremental update: Q_11 = {Q_incremental:.6f}")
    print(f"Closed-form:        Q_11 = {Q_closed:.6f}")
    print(f"Difference:                {abs(Q_incremental - Q_closed):.2e}")

    # Show weights
    print(f"\nWeight on Q_1: (1-α)^{n} = {(1-alpha)**n:.6f}")
    total_weight = (1 - alpha)**n
    for i in range(n):
        w = alpha * (1 - alpha)**(n - 1 - i)
        total_weight += w
        print(f"Weight on R_{i+1}: α(1-α)^{n-1-i} = {w:.6f}")
    print(f"Total weights: {total_weight:.6f}")
