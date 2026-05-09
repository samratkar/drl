---
tags : [chapter2, problems]
categories : drl-problems
subcategories : mab
layout: post
---
# Chapter 2: Multi-Armed Bandits — Numerical Solutions

*Based on Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.), Chapter 2*

---

### N1. Sample-Average Computation

$Q_4(a) = \frac{2 + 4 + 3}{3} = 3.0$

---

### N2. Incremental Update — Step by Step

- $Q_2 = 0 + \frac{1}{1}(4 - 0) = 4.0$
- $Q_3 = 4 + \frac{1}{2}(6 - 4) = 5.0$
- $Q_4 = 5 + \frac{1}{3}(2 - 5) = 4.0$

Verify: $(4 + 6 + 2)/3 = 4.0$ ✓

---

### N3. Epsilon-Greedy Probabilities

- Explore probability per arm: $\epsilon / k = 0.2 / 5 = 0.04$
- Arms 2 and 3 are both greedy. Greedy probability split: $(1 - \epsilon) / 2 = 0.8 / 2 = 0.4$
- $P(\text{arm 1}) = 0.04$
- $P(\text{arm 2}) = 0.4 + 0.04 = 0.44$
- $P(\text{arm 3}) = 0.4 + 0.04 = 0.44$
- $P(\text{arm 4}) = 0.04$
- $P(\text{arm 5}) = 0.04$
- Check: $0.04 + 0.44 + 0.44 + 0.04 + 0.04 = 1.0$ ✓

---

### N4. Constant Step-Size Update

- $Q_2 = 0 + 0.2(5 - 0) = 1.0$
- $Q_3 = 1.0 + 0.2(3 - 1.0) = 1.4$
- $Q_4 = 1.4 + 0.2(7 - 1.4) = 2.52$

---

### N5. Exponential Recency-Weighted Average — Weight Computation

Weight on $R_i$ = $\alpha(1-\alpha)^{n-i} = 0.3 \times 0.7^{5-i}$

| Reward | Weight | Value |
|--------|--------|-------|
| $R_1$ | $0.3 \times 0.7^4 = 0.3 \times 0.2401$ | $0.0720$ |
| $R_2$ | $0.3 \times 0.7^3 = 0.3 \times 0.343$ | $0.1029$ |
| $R_3$ | $0.3 \times 0.7^2 = 0.3 \times 0.49$ | $0.1470$ |
| $R_4$ | $0.3 \times 0.7^1 = 0.3 \times 0.7$ | $0.2100$ |
| $R_5$ | $0.3 \times 0.7^0 = 0.3 \times 1.0$ | $0.3000$ |
| $Q_1$ | $(1-\alpha)^5 = 0.7^5$ | $0.1681$ |
| **Total** | | **1.0000** ✓ |

---

### N6. Verify Weights Sum to 1

Total weight = $(1-\alpha)^n + \sum_{i=1}^n \alpha(1-\alpha)^{n-i}$

$= (1-\alpha)^n + \alpha \sum_{j=0}^{n-1} (1-\alpha)^j$ (substituting $j = n - i$)

$= (1-\alpha)^n + \alpha \cdot \frac{1 - (1-\alpha)^n}{1 - (1-\alpha)}$ (geometric series)

$= (1-\alpha)^n + \alpha \cdot \frac{1 - (1-\alpha)^n}{\alpha}$

$= (1-\alpha)^n + 1 - (1-\alpha)^n = 1$ ✓

---

### N7. Initial Bias Decay

Weight on $Q_1 = (1-\alpha)^n = 0.8^n$

| $n$ | $(0.8)^n$ | Contribution $= 10 \times 0.8^n$ |
|-----|-----------|----------------------------------|
| 1 | 0.800 | 8.00 |
| 3 | 0.512 | 5.12 |
| 5 | 0.328 | 3.28 |
| 10 | 0.107 | 1.07 |
| 20 | 0.012 | 0.12 |

---

### N8. Sample-Average Destroys Initial Bias

- $Q_2 = 100 + \frac{1}{1}(1 - 100) = 1.0$ (initial bias completely gone!)
- $Q_3 = 1.0 + \frac{1}{2}(3 - 1.0) = 2.0$
- $Q_4 = 2.0 + \frac{1}{3}(2 - 2.0) = 2.0$

After just one step, $Q_1 = 100$ has zero influence.

---

### N9. Convergence Conditions Check

| Sequence | $\sum \alpha_n$ | $\sum \alpha_n^2$ | Both satisfied? |
|----------|----------------|-------------------|-----------------|
| $1/n$ | $\infty$ (harmonic) | $\pi^2/6 < \infty$ | **Yes** |
| $0.1$ | $\infty$ | $\infty$ | **No** (2nd fails) |
| $1/n^2$ | $\pi^2/6 < \infty$ | $< \infty$ | **No** (1st fails) |
| $1/\sqrt{n}$ | $\infty$ | $\infty$ ($= \sum 1/n$) | **No** (2nd fails) |

---

### N10. UCB Score Computation

UCB score = $Q_t(a) + 2\sqrt{\ln(100) / N_t(a)}$, where $\ln(100) = 4.605$.

- Arm 1: $2.5 + 2\sqrt{4.605/40} = 2.5 + 2 \times 0.339 = 2.5 + 0.679 = 3.179$
- Arm 2: $1.8 + 2\sqrt{4.605/5} = 1.8 + 2 \times 0.960 = 1.8 + 1.919 = 3.719$
- Arm 3: $3.0 + 2\sqrt{4.605/50} = 3.0 + 2 \times 0.303 = 3.0 + 0.607 = 3.607$

**Arm 2 is selected** (score 3.719) despite having the lowest estimated value, because its uncertainty bonus is huge (only 5 samples).

---

### N11. UCB — Bonus Decay Over Time

Bonus = $2\sqrt{\ln t / (t/2)} = 2\sqrt{2 \ln t / t}$

| $t$ | $\ln t$ | $2\sqrt{2\ln t / t}$ |
|-----|---------|----------------------|
| 10 | 2.303 | $2\sqrt{2 \times 2.303/10} = 2 \times 0.679 = 1.357$ |
| 100 | 4.605 | $2\sqrt{2 \times 4.605/100} = 2 \times 0.303 = 0.607$ |
| 1000 | 6.908 | $2\sqrt{2 \times 6.908/1000} = 2 \times 0.118 = 0.235$ |

The bonus shrinks as the arm is sampled more, transitioning from exploration to exploitation.

---

### N12. Softmax Probabilities from Preferences

- $e^{1.0} = 2.718$
- $e^{2.0} = 7.389$
- $e^{0.5} = 1.649$
- Sum = $2.718 + 7.389 + 1.649 = 11.756$

| Arm | $e^{H_t(a)}$ | $\pi_t(a)$ |
|-----|--------------|-------------|
| 1 | 2.718 | $2.718 / 11.756 = 0.231$ |
| 2 | 7.389 | $7.389 / 11.756 = 0.629$ |
| 3 | 1.649 | $1.649 / 11.756 = 0.140$ |

Check: $0.231 + 0.629 + 0.140 = 1.000$ ✓

---

### N13. Gradient Bandit — One Update Step (With Baseline)

$R_t - \bar{R}_t = 3.5 - 2.0 = 1.5$ (positive — good reward)

- Arm 1 (not selected): $H_{t+1}(1) = 1.0 + 0.1 \times 1.5 \times (0 - 0.231) = 1.0 - 0.035 = 0.965$
- Arm 2 (selected): $H_{t+1}(2) = 2.0 + 0.1 \times 1.5 \times (1 - 0.629) = 2.0 + 0.056 = 2.056$
- Arm 3 (not selected): $H_{t+1}(3) = 0.5 + 0.1 \times 1.5 \times (0 - 0.140) = 0.5 - 0.021 = 0.479$

Arm 2 (selected, good reward) → preference increased. Others decreased.

---

### N14. Gradient Bandit — Bad Reward Update

$R_t - \bar{R}_t = 0.5 - 2.0 = -1.5$ (negative — bad reward)

- Arm 1 (selected): $H_{t+1}(1) = 1.0 + 0.1 \times (-1.5) \times (1 - 0.231) = 1.0 - 0.115 = 0.885$
- Arm 2 (not selected): $H_{t+1}(2) = 2.0 + 0.1 \times (-1.5) \times (0 - 0.629) = 2.0 + 0.094 = 2.094$
- Arm 3 (not selected): $H_{t+1}(3) = 0.5 + 0.1 \times (-1.5) \times (0 - 0.140) = 0.5 + 0.021 = 0.521$

Arm 1 (selected, bad reward) → preference decreased. Others increased (probability shifts away from the bad arm).

---

### N15. Gradient Bandit — Without Baseline

$R_t - \bar{R}_t = 3.8 - 0 = 3.8$ (positive, even though arm 1 might be bad!)

- Arm 1 (selected): $H_{t+1}(1) = 0 + 0.4 \times 3.8 \times (1 - 1/3) = 0 + 1.013 = 1.013$
- Arm 2: $H_{t+1}(2) = 0 + 0.4 \times 3.8 \times (0 - 1/3) = -0.507$
- Arm 3: $H_{t+1}(3) = 0 + 0.4 \times 3.8 \times (0 - 1/3) = -0.507$

Arm 1's preference increased massively — but $R_t = 3.8$ might be below the true mean of arm 1 ($q_{\ast}(1) = 3.5$)! Without a baseline, the algorithm can't tell.

---

### N16. Expected Reward Computation

$\mathbb{E}[R_t] = 0.5 \times 1.0 + 0.3 \times 3.0 + 0.2 \times 2.0 = 0.5 + 0.9 + 0.4 = 1.8$

Optimal: put all probability on arm 2 → $\mathbb{E}[R_t] = 3.0$

The current policy loses $3.0 - 1.8 = 1.2$ in expected reward per step.

---

### N17. Optimistic Greedy — First Few Steps

$Q_2(1) = 5 + 0.1(1.2 - 5) = 5 - 0.38 = 4.62$
$Q_2(2) = 5$ (unchanged)
$Q_2(3) = 5$ (unchanged)

Next step: greedy selects $\argmax[4.62, 5, 5]$ → arm 2 or 3 (tied at 5). The agent moves **away** from arm 1 because $R_1 = 1.2$ was disappointing relative to $Q_1 = 5$. This is the optimistic exploration mechanism in action.

---

### N18. Comparing Step-Sizes — After 5 Rewards

**(a) Sample average:**
$Q_6 = (3 + 1 + 4 + 1 + 5) / 5 = 14/5 = 2.8$

**(b) Constant $\alpha = 0.3$, starting from $Q_1 = 0$:**
- $Q_2 = 0 + 0.3(3 - 0) = 0.9$
- $Q_3 = 0.9 + 0.3(1 - 0.9) = 0.93$
- $Q_4 = 0.93 + 0.3(4 - 0.93) = 1.851$
- $Q_5 = 1.851 + 0.3(1 - 1.851) = 1.596$
- $Q_6 = 1.596 + 0.3(5 - 1.596) = 2.617$

Note $Q_6$ is similar but not identical — the constant $\alpha$ method gives more weight to the recent reward ($R_5 = 5$) and less to the older ones.

---

### N19. UCB — Effect of $c$ Parameter

$\ln(50) = 3.912$. Bonus factor = $\sqrt{3.912/10} = 0.626$.

| $c$ | Bonus = $c \times 0.626$ | Score |
|-----|--------------------------|-------|
| 0.5 | 0.313 | 2.313 |
| 1.0 | 0.626 | 2.626 |
| 2.0 | 1.251 | 3.251 |
| 4.0 | 2.502 | 4.502 |

Larger $c$ → more exploration. At $c = 4$, the bonus dominates the estimated value.

---

### N20. Full Update Comparison — All Methods

**$\epsilon$-greedy**: Random number 0.95 > $\epsilon = 0.1$, so exploit: $\argmax[2.0, 1.5, 2.5] = \textbf{arm 3}$

**UCB** ($c = 2$, $\ln(10) = 2.303$):
- Arm 1: $2.0 + 2\sqrt{2.303/5} = 2.0 + 1.357 = 3.357$
- Arm 2: $1.5 + 2\sqrt{2.303/3} = 1.5 + 1.753 = 3.253$
- Arm 3: $2.5 + 2\sqrt{2.303/2} = 2.5 + 2.147 = 4.647$
- Selected: **arm 3** (highest score — best value AND high uncertainty)

**Gradient bandit** ($H_t = [0.5, 0.3, 0.8]$):
- $e^{0.5} = 1.649$, $e^{0.3} = 1.350$, $e^{0.8} = 2.226$. Sum = 5.225
- $\pi_t = [0.316, 0.258, 0.426]$
- Sample from this distribution. **Arm 3 is most likely** (42.6%) but not guaranteed.

**Greedy** (optimistic or plain): $\argmax[2.0, 1.5, 2.5] = \textbf{arm 3}$

All methods agree here — arm 3 has the highest value *and* the fewest samples (highest uncertainty).
