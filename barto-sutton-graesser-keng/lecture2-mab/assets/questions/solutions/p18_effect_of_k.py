"""P18. Effect of k (Number of Arms)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k=10, epsilon=0.1):
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


if __name__ == "__main__":
    runs, steps = 2000, 2000
    k_values = [5, 10, 20, 50, 100]

    final_optimal = []
    final_reward = []

    for k in k_values:
        b = Bandit(k=k, epsilon=0.1)
        opt_count = 0
        rew_sum = 0
        for run in tqdm(range(runs), desc=f"k={k}"):
            b.reset()
            for t in range(steps):
                a = b.act()
                r = b.step(a)
                if t == steps - 1:
                    rew_sum += r
                    if a == b.best_action:
                        opt_count += 1
        final_optimal.append(opt_count / runs * 100)
        final_reward.append(rew_sum / runs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_values, final_optimal, 'o-', color='#4dabf7', linewidth=2,
             markersize=8)
    ax1.set_xlabel("Number of arms (k)")
    ax1.set_ylabel("% Optimal action at step 2000")
    ax1.set_title("Optimal Action Rate vs k")
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, final_reward, 's-', color='#e8590c', linewidth=2,
             markersize=8)
    ax2.set_xlabel("Number of arms (k)")
    ax2.set_ylabel("Average reward at step 2000")
    ax2.set_title("Reward vs k")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Effect of Number of Arms (ε-greedy, ε=0.1)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("p18_effect_of_k.png", dpi=150)
    plt.close()
    print("Saved: p18_effect_of_k.png")
    for k, opt, rew in zip(k_values, final_optimal, final_reward):
        print(f"  k={k:>3}: {opt:.1f}% optimal, avg reward={rew:.3f}")
