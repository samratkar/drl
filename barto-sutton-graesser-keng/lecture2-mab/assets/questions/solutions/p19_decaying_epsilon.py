"""P19. Decaying Epsilon Schedule"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0.1, epsilon_decay_tau=None):
        self.k = k
        self.epsilon_fixed = epsilon
        self.epsilon_decay_tau = epsilon_decay_tau

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.t = 0

    def get_epsilon(self):
        if self.epsilon_decay_tau is not None:
            return 1.0 / (1 + self.t / self.epsilon_decay_tau)
        return self.epsilon_fixed

    def act(self):
        if np.random.rand() < self.get_epsilon():
            return np.random.randint(self.k)
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
    configs = [
        (Bandit(epsilon=0.1), "Fixed ε=0.1"),
        (Bandit(epsilon_decay_tau=100), "Decay τ=100"),
        (Bandit(epsilon_decay_tau=500), "Decay τ=500"),
        (Bandit(epsilon_decay_tau=2000), "Decay τ=2000"),
    ]

    rewards = np.zeros((len(configs), runs, steps))
    optimal = np.zeros_like(rewards)

    for i, (b, label) in enumerate(configs):
        for run in tqdm(range(runs), desc=label):
            b.reset()
            for t in range(steps):
                a = b.act()
                rewards[i, run, t] = b.step(a)
                if a == b.best_action:
                    optimal[i, run, t] = 1

    rewards = rewards.mean(axis=1)
    optimal = optimal.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for i, (_, label) in enumerate(configs):
        ax1.plot(rewards[i], label=label, linewidth=1.5)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.set_title("Decaying ε vs Fixed ε", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, (_, label) in enumerate(configs):
        ax2.plot(optimal[i] * 100, label=label, linewidth=1.5)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("p19_decaying_epsilon.png", dpi=150)
    plt.close()
    print("Saved: p19_decaying_epsilon.png")
