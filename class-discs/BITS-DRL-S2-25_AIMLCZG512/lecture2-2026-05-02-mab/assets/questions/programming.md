---
tags : [chapter2, problems]
categories : drl-problems
subcategories : mab
---

# Chapter 2: Multi-Armed Bandits — Programming Questions

*Based on Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.), Chapter 2*
*Use Python with numpy. Matplotlib for plotting. Each question is self-contained.*

---

### P1. Implement a Simple Bandit Environment
Write a `BanditEnv` class for a k-armed bandit with:
- Constructor takes `k` (number of arms). True values `q_star[a]` sampled from `N(0, 1)`.
- `pull(a)` method returns a reward sampled from `N(q_star[a], 1)`.
- `optimal_action` property returns the arm with the highest true value.

```python
# Skeleton
class BanditEnv:
    def __init__(self, k=10):
        # TODO: sample q_star from N(0,1)
        pass

    def pull(self, a):
        # TODO: return reward ~ N(q_star[a], 1)
        pass

    @property
    def optimal_action(self):
        # TODO: return argmax of q_star
        pass
```

**Test**: Create a 10-armed bandit, pull each arm 1000 times, verify that the sample means are close to `q_star`.

---

### P2. Sample-Average Action-Value Estimation
Implement a function that maintains action-value estimates using the sample-average method. Track `Q[a]` and `N[a]` (count of pulls). After each pull, update:

$$Q[a] \leftarrow Q[a] + \frac{1}{N[a]}(R - Q[a])$$

Run 1000 steps on a 10-armed bandit, always selecting greedily. Print final `Q` vs true `q_star`.

---

### P3. Epsilon-Greedy Agent
Implement an $\epsilon$-greedy agent that:
- With probability $\epsilon$, selects a random arm
- Otherwise, selects `argmax Q[a]` (break ties randomly)

Run 2000 independent 1000-step experiments for $\epsilon \in \{0, 0.01, 0.1\}$. Plot:
- Average reward vs steps (top panel)
- % optimal action vs steps (bottom panel)

Reproduce Figure 2.2 from the book.

---

### P4. Constant Step-Size Update
Modify your agent to use a constant step-size $\alpha$ instead of sample averages:

$$Q[a] \leftarrow Q[a] + \alpha(R - Q[a])$$

Run with $\alpha = 0.1$ and compare against sample-average ($\alpha = 1/n$) on a stationary 10-armed bandit. Plot both learning curves. Which converges faster initially? Which reaches a better final estimate?

---

### P5. Non-Stationary Bandit
Create a non-stationary bandit where each `q_star[a]` takes an independent random walk after each step:

$$q_{\ast}(a) \leftarrow q_{\ast}(a) + \mathcal{N}(0, 0.01)$$

Compare sample-average vs constant $\alpha = 0.1$ over 10,000 steps (2000 runs). Show that constant $\alpha$ significantly outperforms sample-average in the non-stationary case.

---

### P6. Exponential Recency Weights — Verification
For a single arm with $\alpha = 0.2$ and $Q_1 = 0$, apply 10 rewards `R = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]` using the constant step-size update. Then independently compute $Q_{11}$ using the closed-form:

$$Q_{11} = (1-\alpha)^{10} Q_1 + \sum_{i=1}^{10} \alpha(1-\alpha)^{10-i} R_i$$

Verify that both methods give the same result.

---

### P7. Weight Visualization
For $\alpha \in \{0.1, 0.3, 0.5\}$ and $n = 20$ rewards, plot the weights $\alpha(1-\alpha)^{n-i}$ for $i = 1, \ldots, 20$ as bar charts (one subplot per $\alpha$). Show how larger $\alpha$ concentrates weight on more recent rewards.

---

### P8. Optimistic Initial Values
Implement the optimistic initial values experiment (Figure 2.3):
- Optimistic greedy: $Q_1 = 5$, $\epsilon = 0$, $\alpha = 0.1$
- Realistic $\epsilon$-greedy: $Q_1 = 0$, $\epsilon = 0.1$, $\alpha = 0.1$

Run 2000 runs × 1000 steps. Plot % optimal action over time. Identify the crossover point where optimistic greedy surpasses $\epsilon$-greedy.

---

### P9. Initial Bias Decay Curve
For a single arm with $Q_1 = 5$ and $\alpha \in \{0.1, 0.2, 0.4\}$, plot the weight on $Q_1$ (i.e., $(1-\alpha)^n$) for $n = 0$ to $50$. On the same plot, show the sample-average case ($\alpha = 1/n$) where the weight drops to 0 after step 1. Reproduce the initial-bias-decay diagram.

---

### P10. UCB Action Selection
Implement UCB action selection:

$$A_t = \argmax_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]$$

Handle the $N_t(a) = 0$ case (untried arms should be selected first). Compare UCB ($c = 2$) vs $\epsilon$-greedy ($\epsilon = 0.1$) over 1000 steps, 2000 runs. Plot average reward. Reproduce Figure 2.4.

---

### P11. UCB Score Visualization
At each step $t$ from 1 to 200, record the UCB score for each of the 10 arms. Create a heatmap (arms × time) showing how scores evolve. Observe that untried arms have high scores initially, then scores settle as arms are sampled.

---

### P12. Softmax Action Selection
Implement softmax action selection from preferences $H_t(a)$:

$$\pi_t(a) = \frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}$$

Given $H = [1.0, 2.0, 0.5, -0.3]$, compute and print the probabilities. Sample 10,000 actions from this distribution and verify the empirical frequencies match the theoretical probabilities.

---

### P13. Gradient Bandit — Full Implementation
Implement the gradient bandit algorithm with the update rule:

$$H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R}_t)(\mathbb{1}_{A_t=a} - \pi_t(a))$$

Run four configurations (with/without baseline × $\alpha \in \{0.1, 0.4\}$) on a 10-armed bandit with `true_reward = 4`. Plot % optimal action vs steps for all four. Reproduce Figure 2.5.

---

### P14. Gradient Bandit — Preference Trajectory
Run a single gradient bandit experiment (not averaged) for 500 steps. At each step, record the preference $H_t(a)$ for all 10 arms. Plot all 10 preference trajectories on one chart. Color the optimal arm differently. Observe how the optimal arm's preference rises above the others.

---

### P15. Parameter Study
Implement a full parameter sweep (Figure 2.6):
- $\epsilon$-greedy: $\epsilon \in \{2^{-7}, 2^{-6}, \ldots, 2^{-2}\}$
- Gradient bandit: $\alpha \in \{2^{-5}, 2^{-4}, \ldots, 2^{1}\}$
- UCB: $c \in \{2^{-4}, 2^{-3}, \ldots, 2^{2}\}$
- Optimistic greedy: $Q_1 \in \{2^{-2}, 2^{-1}, \ldots, 2^{2}\}$

For each, compute average reward over 1000 steps, 2000 runs. Plot all four curves on one chart with $x$-axis = $\log_2(\text{parameter})$.

---

### P16. Head-to-Head at Best Parameters
From the parameter study (P15), identify the best parameter for each method. Run all four at their best settings on the same testbed. Plot:
- Average reward over time
- % optimal action over time
- Print a summary table: method, best param, avg reward (full run), avg reward (last 100), % optimal (full), % optimal (last 100)

---

### P17. Exploration Behavior Visualization
For a single 10-armed bandit instance, run each method (greedy, $\epsilon$-greedy, UCB, optimistic) for 200 steps. Create a scatter plot where x-axis = step, y-axis = arm selected, colored by method. Show how:
- Greedy locks onto one arm early
- $\epsilon$-greedy randomly tries others
- UCB systematically cycles through under-explored arms
- Optimistic sweeps through all arms then settles

---

### P18. Effect of $k$ (Number of Arms)
Run $\epsilon$-greedy ($\epsilon = 0.1$) on bandits with $k \in \{5, 10, 20, 50, 100\}$. For each $k$, run 2000 independent experiments of 2000 steps. Plot:
- % optimal action at step 2000 vs $k$
- Average reward at step 2000 vs $k$

Does more arms make the problem harder? By how much?

---

### P19. Decaying Epsilon
Implement a decaying $\epsilon$ schedule: $\epsilon_t = \epsilon_0 / (1 + t/\tau)$ where $\epsilon_0 = 1.0$ and $\tau$ is a decay constant. Compare $\tau \in \{100, 500, 2000\}$ against fixed $\epsilon = 0.1$ on the 10-armed testbed. Does decaying $\epsilon$ outperform fixed $\epsilon$?

---

### P20. Method Robustness — Sensitivity Analysis
For each method at its best parameter, measure performance when the parameter is perturbed by $\pm 50\%$. For example, if UCB's best $c = 0.5$, test $c \in \{0.25, 0.5, 0.75\}$. Create a bar chart showing:
- Average reward at best parameter
- Average reward at $-50\%$
- Average reward at $+50\%$

Which method is most robust (least sensitive to parameter misspecification)?
