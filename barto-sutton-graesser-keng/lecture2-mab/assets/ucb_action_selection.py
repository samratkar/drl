#######################################################################
# Reproduces Figure 2.4 from Sutton & Barto (2nd edition, p.36)      #
# UCB action selection (c=2) vs epsilon-greedy (eps=0.1).             #
# Both use sample averages.                                           #
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
    def __init__(self, k_arm=10, epsilon=0., UCB_param=None):
        self.k = k_arm
        self.epsilon = epsilon
        self.UCB_param = UCB_param
        self.indices = np.arange(self.k)

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            ucb_values = self.q_estimation + \
                self.UCB_param * np.sqrt(
                    np.log(self.time + 1) / (self.action_count + 1e-5)
                )
            q_best = np.max(ucb_values)
            return np.random.choice(
                [a for a, q in enumerate(ucb_values) if q == q_best]
            )

        q_best = np.max(self.q_estimation)
        return np.random.choice(
            [a for a, q in enumerate(self.q_estimation) if q == q_best]
        )

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.q_estimation[action] += (
            (1.0 / self.action_count[action])
            * (reward - self.q_estimation[action])
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


def figure_2_4(runs=2000, time=1000):
    ucb_agent = Bandit(UCB_param=2)
    ucb_agent.label = "UCB (c=2)"

    egreedy_agent = Bandit(epsilon=0.1)
    egreedy_agent.label = "ε-greedy (ε=0.1)"

    bandits = [ucb_agent, egreedy_agent]
    best_action_counts, rewards = simulate(runs, time, bandits)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # --- Top: Average Reward (main figure from the book) ---
    ax1 = axes[0]
    ax1.plot(
        rewards[0],
        color="#2f9e44", linewidth=1.5, alpha=0.9,
        label="UCB $c = 2$"
    )
    ax1.plot(
        rewards[1],
        color="#4dabf7", linewidth=1.5, alpha=0.9,
        label="$\\varepsilon$-greedy $\\varepsilon = 0.1$"
    )
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Average reward", fontsize=12)
    ax1.set_title(
        "Figure 2.4: UCB Action Selection on 10-Armed Testbed\n"
        "Both methods use sample averages",
        fontsize=13, fontweight="bold", pad=12
    )
    ax1.set_xlim(0, time)
    ax1.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- Bottom: % Optimal Action ---
    ax2 = axes[1]
    ax2.plot(
        best_action_counts[0] * 100,
        color="#2f9e44", linewidth=1.5, alpha=0.9,
        label="UCB $c = 2$"
    )
    ax2.plot(
        best_action_counts[1] * 100,
        color="#4dabf7", linewidth=1.5, alpha=0.9,
        label="$\\varepsilon$-greedy $\\varepsilon = 0.1$"
    )
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("% Optimal action", fontsize=12)
    ax2.set_title("% Optimal Action Over Time", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, time)
    ax2.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig("images/figure_2_4_ucb.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/figure_2_4_ucb.png")


if __name__ == "__main__":
    figure_2_4()
