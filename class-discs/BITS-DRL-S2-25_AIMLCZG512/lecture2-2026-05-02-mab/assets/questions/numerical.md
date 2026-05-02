# Chapter 2: Multi-Armed Bandits — Numerical Problems

*Based on Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.), Chapter 2*

---

### N1. Sample-Average Computation
An agent pulls arm $a$ three times and receives rewards $R_1 = 2$, $R_2 = 4$, $R_3 = 3$. Compute $Q_4(a)$ using the sample-average method.

---

### N2. Incremental Update — Step by Step
Starting with $Q_1 = 0$, use the incremental update $Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$ to compute $Q_2$, $Q_3$, and $Q_4$ for rewards $R_1 = 4$, $R_2 = 6$, $R_3 = 2$. Verify your answer against the simple average formula.

---

### N3. Epsilon-Greedy Probabilities
A 5-armed bandit uses $\epsilon$-greedy with $\epsilon = 0.2$. Current estimates: $Q_t = [1.5, 2.3, 2.3, 0.8, 1.0]$. Arms 2 and 3 are tied for the best (assume ties broken uniformly). What is the probability of selecting each arm? Verify the probabilities sum to 1.

---

### N4. Constant Step-Size Update
With $\alpha = 0.2$ and $Q_1 = 0$, compute $Q_2$, $Q_3$, $Q_4$ for rewards $R_1 = 5$, $R_2 = 3$, $R_3 = 7$.

---

### N5. Exponential Recency-Weighted Average — Weight Computation
With $\alpha = 0.3$ and $n = 5$ total rewards observed, compute the weight on each reward $R_i$ ($i = 1, \ldots, 5$) and on the initial value $Q_1$. Verify the weights sum to 1.

---

### N6. Verify Weights Sum to 1
Show algebraically that the weights in the exponential recency-weighted average sum to 1. That is, prove:

$$(1-\alpha)^n + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i} = 1$$

---

### N7. Initial Bias Decay
With $Q_1 = 10$, $\alpha = 0.2$, compute the weight on $Q_1$ and its contribution to $Q_{n+1}$ after $n = 1, 3, 5, 10, 20$ steps.

---

### N8. Sample-Average Destroys Initial Bias
With $Q_1 = 100$ (wildly optimistic) and sample-average ($\alpha_n = 1/n$), compute $Q_2$, $Q_3$, $Q_4$ for rewards $R_1 = 1$, $R_2 = 3$, $R_3 = 2$. What happens to the initial bias after the first step?

---

### N9. Convergence Conditions Check
Determine whether each step-size sequence satisfies the convergence conditions $\sum \alpha_n = \infty$ and $\sum \alpha_n^2 < \infty$:

(a) $\alpha_n = 1/n$  
(b) $\alpha_n = 0.1$ (constant)  
(c) $\alpha_n = 1/n^2$  
(d) $\alpha_n = 1/\sqrt{n}$

---

### N10. UCB Score Computation
At step $t = 100$ with $c = 2$, compute the UCB score for each arm and determine which arm is selected:

| Arm | $Q_t(a)$ | $N_t(a)$ |
|-----|-----------|----------|
| 1 | 2.5 | 40 |
| 2 | 1.8 | 5 |
| 3 | 3.0 | 50 |

---

### N11. UCB — Bonus Decay Over Time
For arm $a$ with $c = 2$, compute the UCB bonus at steps $t = 10, 100, 1000$ assuming $N_t(a) = t/2$ (the arm has been selected half the time). What trend do you observe?

---

### N12. Softmax Probabilities from Preferences
Compute the action probabilities $\pi_t(a)$ from preferences $H_t = [1.0, 2.0, 0.5]$ using softmax. Verify the probabilities sum to 1.

---

### N13. Gradient Bandit — One Update Step (With Baseline)
Given: $H_t = [1.0, 2.0, 0.5]$, $\pi_t = [0.231, 0.629, 0.140]$. Agent selects arm 2 ($A_t = 2$), receives $R_t = 3.5$, baseline $\bar{R}_t = 2.0$, $\alpha = 0.1$. Compute $H_{t+1}$ for all arms.

---

### N14. Gradient Bandit — Bad Reward Update
Same setup as N13 ($H_t = [1.0, 2.0, 0.5]$, $\pi_t = [0.231, 0.629, 0.140]$), but now $R_t = 0.5$ (below baseline $\bar{R}_t = 2.0$) and the agent selects arm 1. Compute $H_{t+1}$ for all arms.

---

### N15. Gradient Bandit — Without Baseline
Using preferences $H_t = [0, 0, 0]$ (so $\pi_t = [1/3, 1/3, 1/3]$) and true reward mean shifted to $+4$. Agent selects arm 1 and receives $R_t = 3.8$. Compute the update with $\alpha = 0.4$ and **no baseline** ($\bar{R}_t = 0$). What is problematic about this result?

---

### N16. Expected Reward Computation
An agent's current policy is $\pi_t = [0.5, 0.3, 0.2]$ and true values are $q_* = [1.0, 3.0, 2.0]$. Compute $\mathbb{E}[R_t]$. What would the optimal expected reward be? How much reward per step is the current policy losing?

---

### N17. Optimistic Greedy — First Few Steps
A 3-armed bandit with $Q_1 = [5, 5, 5]$, $\alpha = 0.1$, true values $q_* = [1, 2, 1.5]$. The greedy agent selects arm 1 (all tied, random tiebreak), receives $R_1 = 1.2$. Compute new estimates for all arms and predict which arm is selected next. Explain why.

---

### N18. Comparing Step-Sizes — After 5 Rewards
An arm receives rewards $R = [3, 1, 4, 1, 5]$. Starting from $Q_1 = 0$, compute $Q_6$ using:
(a) Sample average ($\alpha_n = 1/n$)
(b) Constant $\alpha = 0.3$

Compare the two results and explain the difference.

---

### N19. UCB — Effect of $c$ Parameter
At $t = 50$, arm $a$ has $Q_t(a) = 2.0$ and $N_t(a) = 10$. Compute the UCB score for $c = 0.5, 1, 2, 4$. At what value of $c$ does the exploration bonus dominate the estimated value?

---

### N20. Full Update Comparison — All Methods
A 3-armed bandit at step $t = 10$. Current state:

| Arm | $Q_t(a)$ | $N_t(a)$ |
|-----|-----------|----------|
| 1 | 2.0 | 5 |
| 2 | 1.5 | 3 |
| 3 | 2.5 | 2 |

Which arm does each method select? Use:
- $\epsilon$-greedy with $\epsilon = 0.1$ (the random number for the coin flip is 0.95, i.e., no exploration this step)
- UCB with $c = 2$
- Gradient bandit with preferences $H_t = [0.5, 0.3, 0.8]$
- Plain greedy
