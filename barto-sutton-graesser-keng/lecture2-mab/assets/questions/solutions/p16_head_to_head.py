"""P16. Head-to-Head at Best Parameters"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0., initial=0., alpha=0.1,
                 sample_averages=False, ucb_c=None,
                 gradient=False, gradient_baseline=False):
        self.k = k
        self.epsilon = epsilon
        self.initial = initial
        self.alpha = alpha
        self.sample_averages = sample_averages
        self.ucb_c = ucb_c
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline

    def reset(self):
        self.q_true = np.random.randn(self.k)
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


if __name__ == "__main__":
    runs, steps = 2000, 1000

    # Best parameters (from typical parameter study results)
    bandits = [
        Bandit(epsilon=1/16, sample_averages=True),
        Bandit(gradient=True, alpha=0.25, gradient_baseline=True),
        Bandit(ucb_c=0.5, sample_averages=True),
        Bandit(initial=1.0, alpha=0.1),
    ]
    labels = ["ε-greedy (ε=1/16)", "Gradient (α=0.25)",
              "UCB (c=0.5)", "Optimistic (Q₁=1)"]
    colors = ['#4dabf7', '#d6336c', '#2f9e44', '#e8590c']

    rewards = np.zeros((4, runs, steps))
    optimal = np.zeros_like(rewards)

    for i, b in enumerate(bandits):
        for run in tqdm(range(runs), desc=labels[i]):
            b.reset()
            for t in range(steps):
                a = b.act()
                rewards[i, run, t] = b.step(a)
                if a == b.best_action:
                    optimal[i, run, t] = 1

    rewards_avg = rewards.mean(axis=1)
    optimal_avg = optimal.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for i in range(4):
        ax1.plot(rewards_avg[i], label=labels[i], color=colors[i], linewidth=1.5)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.set_title("Head-to-Head at Best Parameters", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for i in range(4):
        ax2.plot(optimal_avg[i] * 100, label=labels[i], color=colors[i], linewidth=1.5)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("p16_head_to_head.png", dpi=150)
    plt.close()

    print("\nSummary:")
    print(f"{'Method':<25} {'Avg Rew':>8} {'Last100':>8} {'%Opt':>6} {'%Last100':>8}")
    print("-" * 60)
    for i in range(4):
        print(f"{labels[i]:<25} {rewards_avg[i].mean():>8.3f} "
              f"{rewards_avg[i, -100:].mean():>8.3f} "
              f"{optimal_avg[i].mean()*100:>6.1f} "
              f"{optimal_avg[i, -100:].mean()*100:>8.1f}")
    print("\nSaved: p16_head_to_head.png")
