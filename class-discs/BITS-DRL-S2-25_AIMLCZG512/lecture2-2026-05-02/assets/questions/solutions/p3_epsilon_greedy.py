"""P3. Epsilon-Greedy Agent — Reproduce Figure 2.2"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0.):
        self.k = k
        self.epsilon = epsilon

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
        self.Q[a] += (1.0 / self.N[a]) * (r - self.Q[a])
        return r


def simulate(runs, steps, bandits):
    rewards = np.zeros((len(bandits), runs, steps))
    optimal = np.zeros_like(rewards)
    for i, b in enumerate(bandits):
        for run in tqdm(range(runs), desc=f"eps={b.epsilon}"):
            b.reset()
            for t in range(steps):
                a = b.act()
                rewards[i, run, t] = b.step(a)
                if a == b.best_action:
                    optimal[i, run, t] = 1
    return rewards.mean(axis=1), optimal.mean(axis=1)


if __name__ == "__main__":
    runs, steps = 2000, 1000
    epsilons = [0, 0.01, 0.1]
    bandits = [Bandit(epsilon=e) for e in epsilons]

    rewards, optimal = simulate(runs, steps, bandits)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for i, e in enumerate(epsilons):
        ax1.plot(rewards[i], label=f"ε = {e}")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, e in enumerate(epsilons):
        ax2.plot(optimal[i] * 100, label=f"ε = {e}")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("p3_epsilon_greedy.png", dpi=150)
    plt.close()
    print("Saved: p3_epsilon_greedy.png")
