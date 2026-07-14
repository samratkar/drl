"""P14. Gradient Bandit — Preference Trajectory (Single Run)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class GradientBandit:
    def __init__(self, k=10, alpha=0.1, true_reward=0.):
        self.k = k
        self.alpha = alpha
        self.true_reward = true_reward

    def reset(self):
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.H = np.zeros(self.k)
        self.avg_reward = 0.
        self.best_action = np.argmax(self.q_true)
        self.t = 0

    def act(self):
        exp_H = np.exp(self.H - np.max(self.H))
        self.pi = exp_H / np.sum(exp_H)
        return np.random.choice(self.k, p=self.pi)

    def step(self, a):
        r = np.random.randn() + self.q_true[a]
        self.t += 1
        self.avg_reward += (r - self.avg_reward) / self.t
        one_hot = np.zeros(self.k)
        one_hot[a] = 1
        self.H += self.alpha * (r - self.avg_reward) * (one_hot - self.pi)
        return r


if __name__ == "__main__":
    steps = 500
    b = GradientBandit(k=10, alpha=0.1)
    b.reset()

    H_history = np.zeros((steps + 1, b.k))
    H_history[0] = b.H.copy()

    for t in range(steps):
        a = b.act()
        b.step(a)
        H_history[t + 1] = b.H.copy()

    plt.figure(figsize=(12, 6))
    for a in range(b.k):
        if a == b.best_action:
            plt.plot(H_history[:, a], linewidth=2.5, color='red',
                     label=f"Arm {a} (OPTIMAL, q*={b.q_true[a]:.2f})")
        else:
            plt.plot(H_history[:, a], linewidth=0.8, alpha=0.6,
                     label=f"Arm {a} (q*={b.q_true[a]:.2f})")

    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Preference H(a)", fontsize=12)
    plt.title("Gradient Bandit: Preference Trajectories (Single Run)",
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=8, loc='upper left', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("p14_preference_trajectory.png", dpi=150)
    plt.close()
    print("Saved: p14_preference_trajectory.png")
