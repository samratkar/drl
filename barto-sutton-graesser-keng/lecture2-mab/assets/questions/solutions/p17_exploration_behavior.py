"""P17. Exploration Behavior Visualization"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, q_true, epsilon=0., initial=0., alpha=0.1,
                 sample_averages=False, ucb_c=None):
        self.k = len(q_true)
        self.q_true = q_true.copy()
        self.epsilon = epsilon
        self.initial = initial
        self.alpha = alpha
        self.sample_averages = sample_averages
        self.ucb_c = ucb_c
        self.Q = np.zeros(self.k) + initial
        self.N = np.zeros(self.k)
        self.t = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        if self.ucb_c is not None:
            ucb = self.Q + self.ucb_c * np.sqrt(
                np.log(self.t + 1) / (self.N + 1e-5))
            return np.argmax(ucb)
        return np.argmax(self.Q)

    def step(self, a):
        r = np.random.randn() + self.q_true[a]
        self.t += 1
        self.N[a] += 1
        if self.sample_averages:
            self.Q[a] += (1.0 / self.N[a]) * (r - self.Q[a])
        else:
            self.Q[a] += self.alpha * (r - self.Q[a])
        return r


if __name__ == "__main__":
    np.random.seed(42)
    k, steps = 10, 200
    q_true = np.random.randn(k)

    methods = {
        'Greedy': dict(epsilon=0., sample_averages=True),
        'ε-greedy (ε=0.1)': dict(epsilon=0.1, sample_averages=True),
        'UCB (c=2)': dict(ucb_c=2., sample_averages=True),
        'Optimistic (Q₁=5)': dict(initial=5., alpha=0.1),
    }
    colors = ['#868e96', '#4dabf7', '#2f9e44', '#e8590c']

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, ((name, kwargs), color) in enumerate(zip(methods.items(), colors)):
        b = Bandit(q_true, **kwargs)
        actions = []
        for t in range(steps):
            a = b.act()
            b.step(a)
            actions.append(a)
        offset = idx * 0.15 - 0.225
        ax.scatter(range(steps), np.array(actions) + offset,
                   s=6, alpha=0.6, color=color, label=name)

    ax.axhline(y=np.argmax(q_true), color='red', linestyle='--', alpha=0.3,
               label=f"True best (arm {np.argmax(q_true)})")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Arm selected", fontsize=12)
    ax.set_yticks(range(k))
    ax.set_title("Exploration Behavior: Which Arms Are Selected Over Time?",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig("p17_exploration_behavior.png", dpi=150)
    plt.close()
    print("Saved: p17_exploration_behavior.png")
