"""P13. Gradient Bandit — Reproduce Figure 2.5"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class GradientBandit:
    def __init__(self, k=10, alpha=0.1, use_baseline=True, true_reward=0.):
        self.k = k
        self.alpha = alpha
        self.use_baseline = use_baseline
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

        baseline = self.avg_reward if self.use_baseline else 0.
        one_hot = np.zeros(self.k)
        one_hot[a] = 1
        self.H += self.alpha * (r - baseline) * (one_hot - self.pi)
        return r


if __name__ == "__main__":
    runs, steps = 2000, 1000
    configs = [
        (0.1, True,  "α=0.1, with baseline"),
        (0.1, False, "α=0.1, without baseline"),
        (0.4, True,  "α=0.4, with baseline"),
        (0.4, False, "α=0.4, without baseline"),
    ]

    optimal = np.zeros((4, runs, steps))

    for i, (alpha, baseline, label) in enumerate(configs):
        b = GradientBandit(alpha=alpha, use_baseline=baseline, true_reward=4)
        for run in tqdm(range(runs), desc=label):
            b.reset()
            for t in range(steps):
                a = b.act()
                b.step(a)
                if a == b.best_action:
                    optimal[i, run, t] = 1
    optimal = optimal.mean(axis=1)

    plt.figure(figsize=(10, 6))
    for i, (_, _, label) in enumerate(configs):
        plt.plot(optimal[i] * 100, label=label)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.title("Figure 2.5: Gradient Bandit (true reward = +4)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("p13_gradient_bandit.png", dpi=150)
    plt.close()
    print("Saved: p13_gradient_bandit.png")
