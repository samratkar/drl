"""P5. Non-Stationary Bandit — Sample Average vs Constant Alpha"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class NonstationaryBandit:
    def __init__(self, k=10, epsilon=0.1, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

    def reset(self):
        self.q_true = np.zeros(self.k)
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)

    @property
    def best_action(self):
        return np.argmax(self.q_true)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        q_best = np.max(self.Q)
        return np.random.choice(np.where(self.Q == q_best)[0])

    def step(self, a):
        r = np.random.randn() + self.q_true[a]
        self.N[a] += 1
        if self.alpha is not None:
            self.Q[a] += self.alpha * (r - self.Q[a])
        else:
            self.Q[a] += (1.0 / self.N[a]) * (r - self.Q[a])
        self.q_true += np.random.randn(self.k) * 0.01
        return r


if __name__ == "__main__":
    runs, steps = 2000, 10000
    bandits = [
        NonstationaryBandit(epsilon=0.1, alpha=None),
        NonstationaryBandit(epsilon=0.1, alpha=0.1),
    ]
    labels = ["Sample average (α=1/n)", "Constant α=0.1"]

    rewards = np.zeros((2, runs, steps))
    optimal = np.zeros_like(rewards)

    for i, b in enumerate(bandits):
        for run in tqdm(range(runs), desc=labels[i]):
            b.reset()
            for t in range(steps):
                a = b.act()
                if a == b.best_action:
                    optimal[i, run, t] = 1
                rewards[i, run, t] = b.step(a)

    rewards = rewards.mean(axis=1)
    optimal = optimal.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for i, label in enumerate(labels):
        ax1.plot(rewards[i], label=label, linewidth=0.8)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.set_title("Non-Stationary 10-Armed Testbed")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, label in enumerate(labels):
        ax2.plot(optimal[i] * 100, label=label, linewidth=0.8)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("p5_nonstationary.png", dpi=150)
    plt.close()
    print("Saved: p5_nonstationary.png")
