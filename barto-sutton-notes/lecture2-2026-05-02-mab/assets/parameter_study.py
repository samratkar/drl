#######################################################################
# Reproduces Figure 2.6 from Sutton & Barto (2nd edition, p.42)      #
# Parameter study comparing all Chapter 2 methods:                    #
#   - Epsilon-greedy (sample averages)                                #
#   - Gradient bandit (with baseline)                                 #
#   - UCB                                                             #
#   - Optimistic greedy (constant alpha)                              #
#                                                                     #
# Additionally produces:                                              #
#   - Head-to-head comparison of all methods at best parameters       #
#   - Summary statistics table                                        #
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
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1,
                 sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0
        self.average_reward = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            ucb_est = self.q_estimation + \
                self.UCB_param * np.sqrt(
                    np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(ucb_est)
            return np.random.choice(
                [a for a, q in enumerate(ucb_est) if q == q_best])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice(
            [a for a, q in enumerate(self.q_estimation) if q == q_best])

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward += (reward - self.average_reward) / self.time
        self.action_count[action] += 1

        if self.sample_averages:
            self.q_estimation[action] += (
                (1.0 / self.action_count[action])
                * (reward - self.q_estimation[action]))
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            baseline = self.average_reward if self.gradient_baseline else 0
            self.q_estimation += (
                self.step_size * (reward - baseline)
                * (one_hot - self.action_prob))
        else:
            self.q_estimation[action] += (
                self.step_size * (reward - self.q_estimation[action]))
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


# ── Figure 1: Parameter Study (Figure 2.6) ──────────────────────────

def figure_parameter_study(runs=2000, time=1000):
    labels = ['ε-greedy', 'Gradient bandit',
              'UCB', 'Optimistic greedy']
    generators = [
        lambda eps: Bandit(epsilon=eps, sample_averages=True),
        lambda alpha: Bandit(gradient=True, step_size=alpha,
                             gradient_baseline=True),
        lambda c: Bandit(epsilon=0, UCB_param=c, sample_averages=True),
        lambda q0: Bandit(epsilon=0, initial=q0, step_size=0.1),
    ]
    parameters = [
        np.arange(-7, -1, dtype=float),
        np.arange(-5, 2, dtype=float),
        np.arange(-4, 3, dtype=float),
        np.arange(-2, 3, dtype=float),
    ]
    colors = ['#4dabf7', '#d6336c', '#2f9e44', '#e8590c']

    bandits = []
    for gen, params in zip(generators, parameters):
        for p in params:
            b = gen(pow(2, p))
            b.label = f"param={p}"
            bandits.append(b)

    _, avg_rewards = simulate(runs, time, bandits)
    mean_rewards = np.mean(avg_rewards, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    i = 0
    best_params = {}
    for label, params, color in zip(labels, parameters, colors):
        n = len(params)
        segment = mean_rewards[i:i+n]
        ax.plot(params, segment, 'o-', label=label,
                color=color, linewidth=2, markersize=5)
        best_idx = np.argmax(segment)
        best_params[label] = {
            'param_exp': params[best_idx],
            'param_val': pow(2, params[best_idx]),
            'reward': segment[best_idx],
        }
        i += n

    ax.set_xlabel('Parameter ($2^x$)', fontsize=12)
    ax.set_ylabel('Average reward over first 1000 steps', fontsize=12)
    ax.set_title('Figure 2.6: Parameter Study — All Chapter 2 Methods',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/figure_2_6_parameter_study.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: images/figure_2_6_parameter_study.png")
    return best_params


# ── Figure 2: Head-to-head at best parameters ───────────────────────

def figure_head_to_head(best_params, runs=2000, time=1000):
    eg_eps = best_params['ε-greedy']['param_val']
    gb_alpha = best_params['Gradient bandit']['param_val']
    ucb_c = best_params['UCB']['param_val']
    opt_q0 = best_params['Optimistic greedy']['param_val']

    egreedy = Bandit(epsilon=eg_eps, sample_averages=True)
    egreedy.label = f"ε-greedy (ε={eg_eps:.4f})"

    gradient = Bandit(gradient=True, step_size=gb_alpha,
                      gradient_baseline=True)
    gradient.label = f"Gradient bandit (α={gb_alpha:.3f})"

    ucb = Bandit(UCB_param=ucb_c, sample_averages=True)
    ucb.label = f"UCB (c={ucb_c:.2f})"

    optimistic = Bandit(epsilon=0, initial=opt_q0, step_size=0.1)
    optimistic.label = f"Optimistic greedy (Q₁={opt_q0:.2f})"

    bandits = [egreedy, gradient, ucb, optimistic]
    colors = ['#4dabf7', '#d6336c', '#2f9e44', '#e8590c']

    best_action_counts, rewards = simulate(runs, time, bandits)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # ── Top: Average Reward ──
    ax1 = axes[0]
    for i, (bandit, color) in enumerate(zip(bandits, colors)):
        ax1.plot(rewards[i], color=color, linewidth=1.5, alpha=0.9,
                 label=bandit.label)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Average reward', fontsize=12)
    ax1.set_title(
        'Head-to-Head: All Methods at Their Best Parameters',
        fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlim(0, time)
    ax1.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ── Bottom: % Optimal Action ──
    ax2 = axes[1]
    for i, (bandit, color) in enumerate(zip(bandits, colors)):
        ax2.plot(best_action_counts[i] * 100, color=color,
                 linewidth=1.5, alpha=0.9, label=bandit.label)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('% Optimal action', fontsize=12)
    ax2.set_title(
        '% Optimal Action Over Time',
        fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, time)
    ax2.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig('images/figure_2_6_head_to_head.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: images/figure_2_6_head_to_head.png")

    # Return final stats for summary table
    stats = {}
    for i, bandit in enumerate(bandits):
        stats[bandit.label] = {
            'avg_reward_1000': np.mean(rewards[i]),
            'avg_reward_last_100': np.mean(rewards[i, -100:]),
            'optimal_pct_1000': np.mean(best_action_counts[i]) * 100,
            'optimal_pct_last_100': np.mean(best_action_counts[i, -100:]) * 100,
        }
    return stats


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Part 1: Parameter Study (Figure 2.6)")
    print("=" * 60)
    best_params = figure_parameter_study()

    print("\nBest parameter for each method:")
    for method, info in best_params.items():
        print(f"  {method}: 2^{info['param_exp']:.0f} = {info['param_val']:.4f}"
              f"  (avg reward = {info['reward']:.3f})")

    print("\n" + "=" * 60)
    print("Part 2: Head-to-Head at Best Parameters")
    print("=" * 60)
    stats = figure_head_to_head(best_params)

    print("\nSummary Statistics:")
    print(f"{'Method':<40} {'Avg Reward':>11} {'Last-100':>9} "
          f"{'% Optimal':>10} {'Last-100%':>10}")
    print("-" * 82)
    for method, s in stats.items():
        print(f"{method:<40} {s['avg_reward_1000']:>11.3f} "
              f"{s['avg_reward_last_100']:>9.3f} "
              f"{s['optimal_pct_1000']:>10.1f} "
              f"{s['optimal_pct_last_100']:>10.1f}")
