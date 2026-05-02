"""P8. Optimistic Initial Values — Reproduce Figure 2.3"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0., initial=0., alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.initial = initial
        self.alpha = alpha

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.Q = np.zeros(self.k) + self.initial
        self.best_action = np.argmax(self.q_true)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        q_best = np.max(self.Q)
        return np.random.choice(np.where(self.Q == q_best)[0])

    def step(self, a):
        r = np.random.randn() + self.q_true[a]
        self.Q[a] += self.alpha * (r - self.Q[a])
        return r


if __name__ == "__main__":
    runs, steps = 2000, 1000
    bandits = [
        Bandit(epsilon=0, initial=5, alpha=0.1),
        Bandit(epsilon=0.1, initial=0, alpha=0.1),
    ]
    labels = ["Optimistic greedy (Q₁=5, ε=0)", "Realistic ε-greedy (Q₁=0, ε=0.1)"]

    optimal = np.zeros((2, runs, steps))
    for i, b in enumerate(bandits):
        for run in tqdm(range(runs), desc=labels[i]):
            b.reset()
            for t in range(steps):
                a = b.act()
                b.step(a)
                if a == b.best_action:
                    optimal[i, run, t] = 1
    optimal = optimal.mean(axis=1)

    crossover = np.argmax(optimal[0] > optimal[1])

    plt.figure(figsize=(10, 6))
    plt.plot(optimal[0] * 100, label=labels[0], color='#e8590c')
    plt.plot(optimal[1] * 100, label=labels[1], color='#4dabf7')
    if crossover > 0:
        plt.axvline(x=crossover, color='gray', linestyle='--', alpha=0.5)
        plt.annotate(f"Crossover ≈ step {crossover}",
                     xy=(crossover, 50), fontsize=9, color='gray')
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.title("Figure 2.3: Optimistic Initial Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("p8_optimistic_initial.png", dpi=150)
    plt.close()
    print(f"Crossover at step {crossover}")
    print("Saved: p8_optimistic_initial.png")
