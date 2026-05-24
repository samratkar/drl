"""P10. UCB Action Selection — Reproduce Figure 2.4"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0., ucb_c=None):
        self.k = k
        self.epsilon = epsilon
        self.ucb_c = ucb_c

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.t = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)

        if self.ucb_c is not None:
            ucb_values = self.Q + self.ucb_c * np.sqrt(
                np.log(self.t + 1) / (self.N + 1e-5))
            return np.argmax(ucb_values)

        q_best = np.max(self.Q)
        return np.random.choice(np.where(self.Q == q_best)[0])

    def step(self, a):
        r = np.random.randn() + self.q_true[a]
        self.t += 1
        self.N[a] += 1
        self.Q[a] += (1.0 / self.N[a]) * (r - self.Q[a])
        return r


if __name__ == "__main__":
    runs, steps = 2000, 1000
    bandits = [
        Bandit(ucb_c=2),
        Bandit(epsilon=0.1),
    ]
    labels = ["UCB c=2", "ε-greedy ε=0.1"]

    rewards = np.zeros((2, runs, steps))
    for i, b in enumerate(bandits):
        for run in tqdm(range(runs), desc=labels[i]):
            b.reset()
            for t in range(steps):
                a = b.act()
                rewards[i, run, t] = b.step(a)
    rewards = rewards.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards[0], label=labels[0], color='#2f9e44')
    plt.plot(rewards[1], label=labels[1], color='#4dabf7')
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Figure 2.4: UCB vs ε-greedy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("p10_ucb.png", dpi=150)
    plt.close()
    print("Saved: p10_ucb.png")
