"""P4. Constant Step-Size Update vs Sample Average"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0.1, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha  # None = sample average

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)

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
        return r


def simulate(runs, steps, bandits, labels):
    rewards = np.zeros((len(bandits), runs, steps))
    for i, b in enumerate(bandits):
        for run in tqdm(range(runs), desc=labels[i]):
            b.reset()
            for t in range(steps):
                a = b.act()
                rewards[i, run, t] = b.step(a)
    return rewards.mean(axis=1)


if __name__ == "__main__":
    runs, steps = 2000, 1000
    bandits = [
        Bandit(epsilon=0.1, alpha=None),
        Bandit(epsilon=0.1, alpha=0.1),
    ]
    labels = ["Sample average (α=1/n)", "Constant α=0.1"]

    rewards = simulate(runs, steps, bandits, labels)

    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        plt.plot(rewards[i], label=label)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Sample Average vs Constant Step-Size (Stationary)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("p4_constant_stepsize.png", dpi=150)
    plt.close()
    print("Saved: p4_constant_stepsize.png")
