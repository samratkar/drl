#######################################################################
# Reproduces Figure 2.3 from Sutton & Barto (2nd edition, p.34)      #
# Optimistic greedy (Q1=5, eps=0) vs Realistic epsilon-greedy         #
# (Q1=0, eps=0.1). Both use constant step-size alpha=0.1.            #
#                                                                     #
# Based on ten_armed_testbed.py                                       #
# Original copyright:                                                 #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1):
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        self.step_size = step_size
        self.indices = np.arange(self.k)

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        q_best = np.max(self.q_estimation)
        return np.random.choice(
            [a for a, q in enumerate(self.q_estimation) if q == q_best]
        )

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.q_estimation[action] += self.step_size * (
            reward - self.q_estimation[action]
        )
        return reward


def simulate(runs, time, bandits):
    best_action_counts = np.zeros((len(bandits), runs, time))
    rewards = np.zeros_like(best_action_counts)
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs), desc=bandit.label):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    return best_action_counts.mean(axis=1), rewards.mean(axis=1)


def figure_2_3(runs=2000, time=1000):
    optimistic_greedy = Bandit(epsilon=0, initial=5, step_size=0.1)
    optimistic_greedy.label = "Optimistic greedy (Q₁=5, ε=0, α=0.1)"

    realistic_egreedy = Bandit(epsilon=0.1, initial=0, step_size=0.1)
    realistic_egreedy.label = "Realistic ε-greedy (Q₁=0, ε=0.1, α=0.1)"

    bandits = [optimistic_greedy, realistic_egreedy]
    best_action_counts, rewards = simulate(runs, time, bandits)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # --- Top: % Optimal Action (main figure from the book) ---
    ax1 = axes[0]
    ax1.plot(
        best_action_counts[0] * 100,
        color="#e8590c", linewidth=1.5, alpha=0.9,
        label="Optimistic, greedy\n$Q_1 = 5,\\ \\varepsilon = 0$"
    )
    ax1.plot(
        best_action_counts[1] * 100,
        color="#4dabf7", linewidth=1.5, alpha=0.9,
        label="Realistic, $\\varepsilon$-greedy\n$Q_1 = 0,\\ \\varepsilon = 0.1$"
    )
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("% Optimal action", fontsize=12)
    ax1.set_title(
        "Figure 2.3: Optimistic Initial Values on 10-Armed Testbed\n"
        "Both methods use constant step-size α = 0.1",
        fontsize=13, fontweight="bold", pad=12
    )
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, time)
    ax1.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Annotate the crossover region
    crossover_idx = np.argmax(best_action_counts[0] > best_action_counts[1])
    if crossover_idx > 0:
        ax1.axvline(x=crossover_idx, color="gray", linestyle="--", alpha=0.4)
        ax1.annotate(
            f"Crossover ≈ step {crossover_idx}",
            xy=(crossover_idx, 50), fontsize=9, color="gray",
            ha="left", va="center",
            xytext=(crossover_idx + 40, 45),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    # --- Bottom: Average Reward ---
    ax2 = axes[1]
    ax2.plot(
        rewards[0],
        color="#e8590c", linewidth=1.5, alpha=0.9,
        label="Optimistic, greedy\n$Q_1 = 5,\\ \\varepsilon = 0$"
    )
    ax2.plot(
        rewards[1],
        color="#4dabf7", linewidth=1.5, alpha=0.9,
        label="Realistic, $\\varepsilon$-greedy\n$Q_1 = 0,\\ \\varepsilon = 0.1$"
    )
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("Average reward", fontsize=12)
    ax2.set_title("Average Reward Over Time", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(0, time)
    ax2.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig("images/figure_2_3_optimistic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/figure_2_3_optimistic.png")


if __name__ == "__main__":
    figure_2_3()
