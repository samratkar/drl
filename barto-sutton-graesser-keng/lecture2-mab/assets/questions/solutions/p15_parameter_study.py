"""P15. Parameter Study — Reproduce Figure 2.6"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0., initial=0., alpha=0.1,
                 sample_averages=False, ucb_c=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k
        self.epsilon = epsilon
        self.initial = initial
        self.alpha = alpha
        self.sample_averages = sample_averages
        self.ucb_c = ucb_c
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.true_reward = true_reward

    def reset(self):
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.Q = np.zeros(self.k) + self.initial
        self.N = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.t = 0
        self.avg_reward = 0.

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        if self.ucb_c is not None:
            ucb = self.Q + self.ucb_c * np.sqrt(
                np.log(self.t + 1) / (self.N + 1e-5))
            return np.argmax(ucb)
        if self.gradient:
            exp_H = np.exp(self.Q - np.max(self.Q))
            self.pi = exp_H / np.sum(exp_H)
            return np.random.choice(self.k, p=self.pi)
        return np.argmax(self.Q)

    def step(self, a):
        r = np.random.randn() + self.q_true[a]
        self.t += 1
        self.avg_reward += (r - self.avg_reward) / self.t
        self.N[a] += 1
        if self.sample_averages:
            self.Q[a] += (1.0 / self.N[a]) * (r - self.Q[a])
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[a] = 1
            baseline = self.avg_reward if self.gradient_baseline else 0
            self.Q += self.alpha * (r - baseline) * (one_hot - self.pi)
        else:
            self.Q[a] += self.alpha * (r - self.Q[a])
        return r


def simulate(runs, steps, bandits):
    rewards = np.zeros((len(bandits), runs, steps))
    for i, b in enumerate(bandits):
        for run in tqdm(range(runs), desc=f"config {i+1}/{len(bandits)}"):
            b.reset()
            for t in range(steps):
                a = b.act()
                rewards[i, run, t] = b.step(a)
    return np.mean(rewards, axis=(1, 2))


if __name__ == "__main__":
    runs, steps = 2000, 1000

    methods = {
        'ε-greedy': (np.arange(-7, -1, dtype=float),
            lambda e: Bandit(epsilon=2**e, sample_averages=True)),
        'Gradient': (np.arange(-5, 2, dtype=float),
            lambda a: Bandit(gradient=True, alpha=2**a, gradient_baseline=True)),
        'UCB': (np.arange(-4, 3, dtype=float),
            lambda c: Bandit(ucb_c=2**c, sample_averages=True)),
        'Optimistic': (np.arange(-2, 3, dtype=float),
            lambda q: Bandit(initial=2**q, alpha=0.1)),
    }
    colors = ['#4dabf7', '#d6336c', '#2f9e44', '#e8590c']

    plt.figure(figsize=(10, 6))
    for (name, (params, gen)), color in zip(methods.items(), colors):
        bandits = [gen(p) for p in params]
        avg_rewards = simulate(runs, steps, bandits)
        plt.plot(params, avg_rewards, 'o-', label=name, color=color,
                 linewidth=2, markersize=5)

    plt.xlabel("Parameter (2^x)", fontsize=12)
    plt.ylabel("Average reward over 1000 steps", fontsize=12)
    plt.title("Figure 2.6: Parameter Study", fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("p15_parameter_study.png", dpi=150)
    plt.close()
    print("Saved: p15_parameter_study.png")
