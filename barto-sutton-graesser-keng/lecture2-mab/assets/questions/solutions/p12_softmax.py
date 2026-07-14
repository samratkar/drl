"""P12. Softmax Action Selection"""
import numpy as np

def softmax(H):
    exp_H = np.exp(H - np.max(H))  # subtract max for numerical stability
    return exp_H / np.sum(exp_H)

if __name__ == "__main__":
    H = np.array([1.0, 2.0, 0.5, -0.3])
    probs = softmax(H)

    print("Preferences H:", H)
    print("Probabilities π:", np.round(probs, 4))
    print("Sum:", np.round(np.sum(probs), 6))

    n_samples = 10000
    actions = np.random.choice(len(H), size=n_samples, p=probs)
    empirical = np.bincount(actions, minlength=len(H)) / n_samples

    print(f"\nEmpirical frequencies ({n_samples} samples):")
    print(f"{'Arm':<5} {'Theoretical':>12} {'Empirical':>10} {'Diff':>8}")
    for a in range(len(H)):
        print(f"{a:<5} {probs[a]:>12.4f} {empirical[a]:>10.4f} "
              f"{abs(probs[a] - empirical[a]):>8.4f}")
