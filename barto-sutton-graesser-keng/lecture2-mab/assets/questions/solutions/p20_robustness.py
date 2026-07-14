"""P20. Method Robustness — Sensitivity Analysis"""
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


def evaluate(bandit, runs=2000, steps=1000):
    total = 0.
    for _ in tqdm(range(runs), leave=False):
        bandit.reset()
        for t in range(steps):
            a = bandit.act()
            total += bandit.step(a)
    return total / (runs * steps)


if __name__ == "__main__":
    # Best parameters and their ±50% perturbations
    methods = {
        'ε-greedy': {
            'best': 1/16,
            'configs': lambda p: Bandit(epsilon=p, sample_averages=True),
        },
        'Gradient': {
            'best': 0.25,
            'configs': lambda p: Bandit(gradient=True, alpha=p,
                                        gradient_baseline=True),
        },
        'UCB': {
            'best': 0.5,
            'configs': lambda p: Bandit(ucb_c=p, sample_averages=True),
        },
        'Optimistic': {
            'best': 1.0,
            'configs': lambda p: Bandit(initial=p, alpha=0.1),
        },
    }

    results = {}
    for name, m in methods.items():
        best = m['best']
        params = [best * 0.5, best, best * 1.5]
        labels_p = [f"-50% ({params[0]:.4f})", f"Best ({params[1]:.4f})",
                    f"+50% ({params[2]:.4f})"]
        rewards = []
        for p, lbl in zip(params, labels_p):
            print(f"  {name} {lbl}...")
            r = evaluate(m['configs'](p))
            rewards.append(r)
        results[name] = rewards

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.25
    colors_bar = ['#868e96', '#2f9e44', '#e8590c']
    bar_labels = ['-50%', 'Best', '+50%']

    for i in range(3):
        vals = [results[m][i] for m in methods]
        ax.bar(x + (i - 1) * width, vals, width, label=bar_labels[i],
               color=colors_bar[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods.keys(), fontsize=11)
    ax.set_ylabel("Average reward", fontsize=12)
    ax.set_title("Robustness: Performance at Best vs ±50% Parameter",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("p20_robustness.png", dpi=150)
    plt.close()

    print("\nResults:")
    print(f"{'Method':<12} {'-50%':>8} {'Best':>8} {'+50%':>8} {'Spread':>8}")
    print("-" * 48)
    for name, r in results.items():
        spread = max(r) - min(r)
        print(f"{name:<12} {r[0]:>8.3f} {r[1]:>8.3f} {r[2]:>8.3f} {spread:>8.3f}")
    print("\nSmallest spread = most robust method")
    print("Saved: p20_robustness.png")
