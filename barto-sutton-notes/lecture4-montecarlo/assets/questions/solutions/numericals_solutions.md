---
layout: post
---

# Numerical Questions & Solutions - Lecture 4 (Monte Carlo Methods)

---

## Question 1: Basic Return Calculation [3 marks]

An agent generates the following episode in a 4-state environment:

![Q1 Diagram](../images/q1-return.svg)

**Episode:** S₁ →(a₁, r=+3)→ S₂ →(a₂, r=−2)→ S₃ →(a₁, r=+6)→ Terminal

**Given:** γ = 0.9

Compute the return $G_t$ from each state, working backwards from the terminal state.

### Solution 1

Working backwards:

| Time | State | Computation | Return G |
|------|-------|-------------|----------|
| t=2 | S₃ | G₂ = r₃ = +6 | **6.0** |
| t=1 | S₂ | G₁ = r₂ + γ·G₂ = −2 + 0.9×6 = −2 + 5.4 | **3.4** |
| t=0 | S₁ | G₀ = r₁ + γ·G₁ = 3 + 0.9×3.4 = 3 + 3.06 | **6.06** |

---

## Question 2: First-Visit vs Every-Visit MC [5 marks]

An agent generates a single episode in a 3-state environment where state B is visited twice:

![Q2 Diagram](../images/q2-firstvisit.svg)

**Episode:** A →(Right, r=+1)→ B →(Up, r=−1)→ A →(Right, r=+2)→ B →(Down, r=+4)→ Terminal

**Given:**
- γ = 1.0 (undiscounted)
- Initial V(s) = 0 for all states

**(a)** [2 marks] List the first-visit return for each state.

**(b)** [2 marks] List the every-visit return for each state (average all visits).

**(c)** [1 mark] Which method gives a lower estimate for V(B)? Why?

### Solution 2

Returns from each time step (γ=1.0 so G = sum of all future rewards):

| Time | State | G = sum of future rewards |
|------|-------|--------------------------|
| t=0 | A | 1 + (−1) + 2 + 4 = **6** |
| t=1 | B | (−1) + 2 + 4 = **5** |
| t=2 | A | 2 + 4 = **6** |
| t=3 | B | 4 = **4** |

**(a)** First-visit returns:
- V(A) = G at t=0 (first visit) = **6**
- V(B) = G at t=1 (first visit) = **5**

**(b)** Every-visit returns:
- V(A) = average(G at t=0, G at t=2) = (6 + 6)/2 = **6**
- V(B) = average(G at t=1, G at t=3) = (5 + 4)/2 = **4.5**

**(c)** Every-visit gives lower V(B) = 4.5 vs first-visit V(B) = 5.
Because the second visit to B (t=3) has a shorter remaining trajectory with less cumulative reward, pulling the average down.

---

## Question 3: On-Policy First-Visit MC for Q-values [5 marks]

An agent following an ε-greedy policy generates three episodes:

![Q3 Diagram](../images/q3-onpolicy-q.svg)

**Episode 1:** S₁ →(Left, r=+2)→ S₂ →(Right, r=+4)→ Terminal
**Episode 2:** S₁ →(Left, r=+2)→ S₂ →(Left, r=−3)→ Terminal
**Episode 3:** S₁ →(Right, r=+7)→ Terminal

**Given:**
- γ = 0.95
- First-visit MC, simple averaging
- Initial Q(s, a) = 0 for all pairs

**(a)** [2 marks] Compute the return $G_0$ for each episode.

**(b)** [2 marks] After processing all 3 episodes, compute Q(S₁, Left) and Q(S₁, Right).

**(c)** [1 mark] Under the greedy policy derived from these Q-values, what action would be selected at S₁?

### Solution 3

**(a)** Returns:

**Episode 1:** S₁ →(Left, r=+2)→ S₂ →(Right, r=+4)→ Terminal
- G₀ = 2 + 0.95×4 = 2 + 3.8 = **5.8**

**Episode 2:** S₁ →(Left, r=+2)→ S₂ →(Left, r=−3)→ Terminal
- G₀ = 2 + 0.95×(−3) = 2 − 2.85 = **−0.85**

**Episode 3:** S₁ →(Right, r=+7)→ Terminal
- G₀ = 7 = **7.0**

**(b)** Q-value estimates:

**Q(S₁, Left):** First visited with action Left in Episodes 1 and 2.
- Q(S₁, Left) = (5.8 + (−0.85)) / 2 = 4.95 / 2 = **2.475**

**Q(S₁, Right):** First visited with action Right in Episode 3 only.
- Q(S₁, Right) = 7.0 / 1 = **7.0**

**(c)** Greedy policy: argmax Q(S₁, a) → **Right** (since 7.0 > 2.475)

---

## Question 4: Constant-α MC Update [4 marks]

An agent uses constant-α MC to update Q-values after each episode.

**Episode 1:** S →(a, r=+10)→ Terminal
**Episode 2:** S →(a, r=+4)→ Terminal
**Episode 3:** S →(a, r=+7)→ Terminal

**Given:**
- γ = 1.0, α = 0.3
- Initial Q(S, a) = 0

**(a)** [2 marks] Compute Q(S, a) after each episode using the constant-α update rule.

**(b)** [2 marks] What would the simple-averaging (first-visit) estimate be after all 3 episodes? Why does it differ from the constant-α result?

### Solution 4

**(a)** Step-by-step updates:

| Episode | G | Q(old) | Error (G−Q) | α×Error | Q(new) |
|---------|---|--------|-------------|---------|--------|
| 1 | 10 | 0 | 10 − 0 = 10 | 0.3×10 = 3.0 | **3.0** |
| 2 | 4 | 3.0 | 4 − 3 = 1 | 0.3×1 = 0.3 | **3.3** |
| 3 | 7 | 3.3 | 7 − 3.3 = 3.7 | 0.3×3.7 = 1.11 | **4.41** |

**(b)** Simple average: (10 + 4 + 7)/3 = 21/3 = **7.0**

The constant-α result (4.41) differs from the simple average (7.0) because:
- Constant-α gives exponentially decaying weight to older returns
- Recent returns are weighted more heavily
- The first return (10) has been "forgotten" — its effective weight is 0.3×(1−0.3)² = 0.147, while the last return's weight is 0.3

---

## Question 5: Importance Sampling Ratio — Single Episode [4 marks]

An agent operates in a 3-state environment. One episode is generated under behavior policy b:

![Q5 Diagram](../images/q5-is-ratio.svg)

**Episode:** S₁ →(a₁, r=+5)→ S₂ →(a₂, r=+3)→ S₃ →(a₁, r=+2)→ Terminal

**Policy probabilities:**

| State | Action taken | π(a|s) | b(a|s) |
|-------|-------------|--------|--------|
| S₁    | a₁          | 0.9    | 0.5    |
| S₂    | a₂          | 0.6    | 0.4    |
| S₃    | a₁          | 1.0    | 0.7    |

**(a)** [2 marks] Compute the per-step importance sampling ratios.

**(b)** [2 marks] Compute the cumulative importance sampling ratio $\rho_{0:2}$ for the full episode starting from S₁.

### Solution 5

**(a)** Per-step ratios:

| Step | State | Action | π(a|s) | b(a|s) | ρ_t = π/b |
|------|-------|--------|--------|--------|-----------|
| t=0 | S₁ | a₁ | 0.9 | 0.5 | 0.9/0.5 = **1.8** |
| t=1 | S₂ | a₂ | 0.6 | 0.4 | 0.6/0.4 = **1.5** |
| t=2 | S₃ | a₁ | 1.0 | 0.7 | 1.0/0.7 = **1.429** |

**(b)** Cumulative ratio:

$\rho_{0:2} = 1.8 \times 1.5 \times 1.429 = 1.8 \times 2.143 = \mathbf{3.857}$

---

## Question 6: Off-Policy MC Prediction with Ordinary IS [5 marks]

An agent generates one episode under behavior policy b:

![Q6 Diagram](../images/q6-offpolicy-ois.svg)

**Episode:** X →(a₁, r=+4)→ Y →(a₂, r=+6)→ Terminal

**Given:**
- γ = 0.9, α = 0.5 (constant-α MC)
- Initial Q(X, a₁) = 2, Q(Y, a₂) = 1
- π(a₁|X) = 1.0, b(a₁|X) = 0.6
- π(a₂|Y) = 0.8, b(a₂|Y) = 0.5

**(a)** [1 mark] Compute the return G from each state.

**(b)** [2 marks] Compute the cumulative importance sampling ratios for each time step.

**(c)** [2 marks] Compute the updated Q-values using off-policy constant-α MC with ordinary importance sampling.

### Solution 6

**(a)** Returns:

| Time | State-Action | Computation | Return G |
|------|-------------|-------------|----------|
| t=1 | (Y, a₂) | G₁ = r₂ = +6 | **6** |
| t=0 | (X, a₁) | G₀ = r₁ + γ·G₁ = 4 + 0.9×6 = 4 + 5.4 | **9.4** |

**(b)** IS ratios:

| Time | π/b (single) | Cumulative ρ |
|------|-------------|--------------|
| t=1 | 0.8/0.5 = 1.6 | **1.6** |
| t=0 | 1.0/0.6 = 1.667 | 1.667 × 1.6 = **2.667** |

**(c)** Off-policy updates:

**Q(Y, a₂):**

| Step | Value |
|------|-------|
| Q_old | 1 |
| G₁ | 6 |
| ρ₁:₁ | 1.6 |
| ρ·G | 1.6 × 6 = 9.6 |
| Error: ρ·G − Q | 9.6 − 1 = 8.6 |
| α × Error | 0.5 × 8.6 = 4.3 |
| **Q_new** | 1 + 4.3 = **5.3** |

**Q(X, a₁):**

| Step | Value |
|------|-------|
| Q_old | 2 |
| G₀ | 9.4 |
| ρ₀:₁ | 2.667 |
| ρ·G | 2.667 × 9.4 = 25.067 |
| Error: ρ·G − Q | 25.067 − 2 = 23.067 |
| α × Error | 0.5 × 23.067 = 11.533 |
| **Q_new** | 2 + 11.533 = **13.533** |

---

## Question 7: Weighted Importance Sampling [5 marks]

An agent generates three episodes starting from state S, all taking action a:

| Episode | Return G | Importance Ratio ρ |
|---------|----------|-------------------|
| 1       | 12       | 2.0               |
| 2       | 6        | 0.5               |
| 3       | 9        | 1.5               |

**Given:** Initial Q(S, a) = 0

**(a)** [2 marks] Compute the ordinary importance sampling estimate of Q(S, a).

**(b)** [2 marks] Compute the weighted importance sampling estimate of Q(S, a).

**(c)** [1 mark] Which estimate has lower variance? Justify briefly.

### Solution 7

**(a)** Ordinary IS estimate:

$\hat{V}_{OIS} = \frac{1}{N}\sum_i \rho_i G_i = \frac{1}{3}(2.0×12 + 0.5×6 + 1.5×9)$

$= \frac{1}{3}(24 + 3 + 13.5) = \frac{40.5}{3} = \mathbf{13.5}$

**(b)** Weighted IS estimate:

$\hat{V}_{WIS} = \frac{\sum_i \rho_i G_i}{\sum_i \rho_i} = \frac{24 + 3 + 13.5}{2.0 + 0.5 + 1.5} = \frac{40.5}{4.0} = \mathbf{10.125}$

**(c)** Ordinary IS is **unbiased** but has higher variance. Weighted IS is **biased** (bias → 0 as N → ∞) but typically has much lower variance because it normalizes by the sum of weights, preventing extreme ratios from dominating the estimate.

---

## Question 8: MC Control — ε-Greedy Policy Improvement [5 marks]

After running MC prediction, an agent has the following Q-table:

| State | Left | Right | Up   |
|-------|------|-------|------|
| S₁    | 4.0  | 7.0   | 2.0  |
| S₂    | 5.5  | 3.0   | 6.0  |

**Given:** ε = 0.2, 3 actions available

**(a)** [2 marks] Compute the ε-greedy policy π(a|S₁) for each action.

**(b)** [2 marks] Compute the ε-greedy policy π(a|S₂) for each action.

**(c)** [1 mark] If ε is decreased to 0, what is the resulting policy? What problem might this cause for MC methods?

### Solution 8

**(a)** Policy at S₁:
Best action = Right (Q = 7.0)
- π(Right|S₁) = 1 − ε + ε/|A| = 1 − 0.2 + 0.2/3 = 0.8 + 0.0667 = **0.867**
- π(Left|S₁) = ε/|A| = 0.2/3 = **0.067**
- π(Up|S₁) = ε/|A| = 0.2/3 = **0.067**

**(b)** Policy at S₂:
Best action = Up (Q = 6.0)
- π(Up|S₂) = 1 − ε + ε/|A| = 0.8 + 0.0667 = **0.867**
- π(Left|S₂) = ε/|A| = **0.067**
- π(Right|S₂) = ε/|A| = **0.067**

**(c)** If ε = 0: Pure greedy policy — π(Right|S₁) = 1, π(Up|S₂) = 1.
**Problem:** The agent stops exploring entirely. MC requires exploration to visit all state-action pairs. Without exploring starts or ε > 0, many Q(s,a) will never be updated, potentially missing better actions.

---

## Question 9: Discounted Returns in a Loop [4 marks]

An agent generates an episode where it loops through a state before reaching terminal:

![Q9 Diagram](../images/q9-loop.svg)

**Episode:** A →(r=+1)→ B →(r=+2)→ A →(r=+1)→ B →(r=+3)→ Terminal

**Given:** γ = 0.8

**(a)** [2 marks] Compute the first-visit return G(A) and G(B).

**(b)** [2 marks] Compute the every-visit returns for A and B (average over all visits).

### Solution 9

Returns from each time step:

| Time | State | Computation | G |
|------|-------|-------------|---|
| t=3 | B | G₃ = 3 | 3 |
| t=2 | A | G₂ = 1 + 0.8×3 = 1 + 2.4 | 3.4 |
| t=1 | B | G₁ = 2 + 0.8×3.4 = 2 + 2.72 | 4.72 |
| t=0 | A | G₀ = 1 + 0.8×4.72 = 1 + 3.776 | 4.776 |

**(a)** First-visit returns:
- **G(A)** = G at t=0 = **4.776**
- **G(B)** = G at t=1 = **4.72**

**(b)** Every-visit returns:
- **V(A)** = (G₀ + G₂)/2 = (4.776 + 3.4)/2 = **4.088**
- **V(B)** = (G₁ + G₃)/2 = (4.72 + 3)/2 = **3.86**

---

## Question 10: Off-Policy MC — Zero Probability Problem [3 marks]

An agent generates an episode under behavior policy b:

**Episode:** S₁ →(a₂, r=+5)→ S₂ →(a₁, r=+8)→ Terminal

**Policy probabilities:**

| State | Action | π(a|s) | b(a|s) |
|-------|--------|--------|--------|
| S₁    | a₂     | 0.0    | 0.4    |
| S₂    | a₁     | 1.0    | 0.6    |

**(a)** [1 mark] Compute the importance sampling ratio ρ₀:₁.

**(b)** [2 marks] What does this result imply about off-policy learning from this episode? Explain intuitively why this makes sense.

### Solution 10

**(a)** IS ratio:

$\rho_{0:1} = \frac{\pi(a_2|S_1)}{b(a_2|S_1)} \times \frac{\pi(a_1|S_2)}{b(a_1|S_2)} = \frac{0.0}{0.4} \times \frac{1.0}{0.6} = 0 \times 1.667 = \mathbf{0}$

**(b)** The IS ratio is zero, meaning this entire episode contributes **nothing** to the off-policy estimate of Q(S₁, a₂).

**Intuition:** The target policy π would never take action a₂ at S₁ (π(a₂|S₁)=0). Therefore, this trajectory is impossible under π, and it makes sense that it contributes zero information about π's value function. This is the "coverage" requirement: off-policy MC can only learn about trajectories that π would actually take.

---

## Question 11: Incremental MC Update Derivation [4 marks]

An agent has processed N=4 episodes for state S, obtaining returns: G₁=5, G₂=8, G₃=3, G₄=10.

**(a)** [2 marks] Show that the sample average after N episodes can be written incrementally as:
$$V_N = V_{N-1} + \frac{1}{N}[G_N - V_{N-1}]$$
by computing V₁, V₂, V₃, V₄ step by step.

**(b)** [2 marks] Verify that V₄ equals the batch average (5+8+3+10)/4.

### Solution 11

**(a)** Incremental computation:

| N | G_N | V_{N-1} | 1/N | G_N − V_{N-1} | V_N |
|---|-----|---------|-----|----------------|-----|
| 1 | 5 | 0 | 1.0 | 5 − 0 = 5 | 0 + 1.0×5 = **5.0** |
| 2 | 8 | 5.0 | 0.5 | 8 − 5 = 3 | 5.0 + 0.5×3 = **6.5** |
| 3 | 3 | 6.5 | 0.333 | 3 − 6.5 = −3.5 | 6.5 + 0.333×(−3.5) = **5.333** |
| 4 | 10 | 5.333 | 0.25 | 10 − 5.333 = 4.667 | 5.333 + 0.25×4.667 = **6.5** |

**(b)** Batch average = (5 + 8 + 3 + 10)/4 = 26/4 = **6.5** ✓

The incremental formula produces the exact same result as the batch average.

---

## Question 12: Comparing MC and TD on Same Episode [5 marks]

An agent generates an episode in a 3-state chain:

![Q12 Diagram](../images/q12-mc-vs-td.svg)

**Episode:** S₁ →(r=+2)→ S₂ →(r=+4)→ S₃ →(r=+1)→ Terminal

**Given:**
- γ = 0.9, α = 0.5
- Initial V(S₁) = 0, V(S₂) = 0, V(S₃) = 0

**(a)** [2 marks] Compute the MC update for each state (use the full return from each state).

**(b)** [2 marks] Compute the TD(0) update for each state (one-step bootstrap).

**(c)** [1 mark] Which method gives a larger update for V(S₁)? Explain in one sentence.

### Solution 12

**(a)** MC updates (full return from each state):

| State | Return G | Update: V ← V + α(G − V) | V_new |
|-------|----------|---------------------------|-------|
| S₃ | G₃ = 1 | 0 + 0.5(1 − 0) = 0.5 | **0.5** |
| S₂ | G₂ = 4 + 0.9×1 = 4.9 | 0 + 0.5(4.9 − 0) = 2.45 | **2.45** |
| S₁ | G₁ = 2 + 0.9×4.9 = 6.41 | 0 + 0.5(6.41 − 0) = 3.205 | **3.205** |

**(b)** TD(0) updates (one-step bootstrap, using initial V values):

| State | TD Target: r + γV(S') | Update: V ← V + α(target − V) | V_new |
|-------|----------------------|-------------------------------|-------|
| S₃ | 1 + 0.9×0 = 1 | 0 + 0.5(1 − 0) = 0.5 | **0.5** |
| S₂ | 4 + 0.9×0 = 4 | 0 + 0.5(4 − 0) = 2.0 | **2.0** |
| S₁ | 2 + 0.9×0 = 2 | 0 + 0.5(2 − 0) = 1.0 | **1.0** |

**(c)** MC gives larger update for V(S₁): 3.205 vs 1.0.

MC propagates information from the entire trajectory back to S₁ in one update, while TD(0) only uses the immediate reward and the (still-zero) estimate of the next state, requiring multiple episodes to propagate information backwards.

---

## Question 13: Off-Policy with Non-Greedy Target Policy [5 marks]

An agent uses a stochastic target policy π (not fully greedy):

**Episode (under b):** S₁ →(a₁, r=+3)→ S₂ →(a₂, r=+5)→ Terminal

**Given:**
- γ = 1.0, α = 0.4
- Initial Q(S₁, a₁) = 0, Q(S₂, a₂) = 0
- π(a₁|S₁) = 0.7, b(a₁|S₁) = 0.5
- π(a₂|S₂) = 0.6, b(a₂|S₂) = 0.8

**(a)** [1 mark] Compute returns from each state.

**(b)** [2 marks] Compute off-policy update for Q(S₂, a₂) using ordinary IS.

**(c)** [2 marks] Compute off-policy update for Q(S₁, a₁) using ordinary IS. Show the cumulative ratio.

### Solution 13

**(a)** Returns:
- G₁ (from S₂) = r₂ = 5 → **G₁ = 5**
- G₀ (from S₁) = r₁ + γ·G₁ = 3 + 1.0×5 = **G₀ = 8**

**(b)** Off-policy update for Q(S₂, a₂):

- ρ₁:₁ = π(a₂|S₂)/b(a₂|S₂) = 0.6/0.8 = **0.75**
- ρ·G = 0.75 × 5 = 3.75
- Error: 3.75 − 0 = 3.75
- Update: 0 + 0.4 × 3.75 = **1.5**
- **Q(S₂, a₂) = 1.5**

**(c)** Off-policy update for Q(S₁, a₁):

- ρ₀:₁ = [π(a₁|S₁)/b(a₁|S₁)] × [π(a₂|S₂)/b(a₂|S₂)] = (0.7/0.5) × (0.6/0.8) = 1.4 × 0.75 = **1.05**
- ρ·G = 1.05 × 8 = 8.4
- Error: 8.4 − 0 = 8.4
- Update: 0 + 0.4 × 8.4 = **3.36**
- **Q(S₁, a₁) = 3.36**

Note: The cumulative ratio is 1.05, only slightly above 1, because the product of a ratio > 1 (1.4) and a ratio < 1 (0.75) partially cancel.

---

## Question 14: Variance of Ordinary vs Weighted IS [5 marks]

Five episodes from state S (action a) produce:

| Episode | Return G | ρ     |
|---------|----------|-------|
| 1       | 10       | 3.0   |
| 2       | 4        | 0.2   |
| 3       | 7        | 1.5   |
| 4       | 2        | 4.0   |
| 5       | 8        | 0.8   |

**(a)** [2 marks] Compute the ordinary IS estimate: $\hat{V}_{OIS} = \frac{1}{N}\sum_{i} \rho_i G_i$

**(b)** [2 marks] Compute the weighted IS estimate: $\hat{V}_{WIS} = \frac{\sum_i \rho_i G_i}{\sum_i \rho_i}$

**(c)** [1 mark] Which estimate is unbiased? Which typically has lower variance?

### Solution 14

**(a)** Ordinary IS estimate:

$\hat{V}_{OIS} = \frac{1}{5}\sum_i \rho_i G_i = \frac{1}{5}(3.0×10 + 0.2×4 + 1.5×7 + 4.0×2 + 0.8×8)$

$= \frac{1}{5}(30 + 0.8 + 10.5 + 8 + 6.4) = \frac{55.7}{5} = \mathbf{11.14}$

**(b)** Weighted IS estimate:

$\hat{V}_{WIS} = \frac{\sum_i \rho_i G_i}{\sum_i \rho_i} = \frac{55.7}{3.0 + 0.2 + 1.5 + 4.0 + 0.8} = \frac{55.7}{9.5} = \mathbf{5.863}$

**(c)** Ordinary IS is **unbiased**. Weighted IS typically has **lower variance**.

The high variance of OIS here is evident: extreme ratios like ρ=4.0 amplify small returns (G=2 becomes 8) and ρ=3.0 amplifies large returns (G=10 becomes 30), causing wild swings. WIS normalizes these away.

---

## Question 15: MC Exploring Starts [4 marks]

In MC with exploring starts, every state-action pair must have nonzero probability of being the starting pair.

Consider a 2-state, 2-action environment. After many episodes, the Q-table is:

| State | Action A | Action B |
|-------|----------|----------|
| S₁    | 4.2      | 4.2      |
| S₂    | 6.0      | 5.8      |

**(a)** [2 marks] What is the greedy policy derived from this Q-table? What issue arises at S₁?

**(b)** [2 marks] If we use ε-greedy with ε=0.1 instead of exploring starts, write the probability of each action at S₁ and S₂.

### Solution 15

**(a)** Greedy policy:
- S₁: Both Left and Right have Q = 4.2 → **tie**. The greedy policy is undefined (or arbitrary tie-breaking).
- S₂: Action A (Q=6.0 > 5.8) → select **A**

**Issue at S₁:** With equal Q-values, greedy arbitrarily picks one action and may never sample the other, leading to the Q-estimate for the unchosen action remaining forever unchanged.

**(b)** ε-greedy with ε=0.1, 2 actions:

**S₁** (tie → either can be "greedy", say A by convention):
- π(A|S₁) = 1 − ε + ε/2 = 0.9 + 0.05 = **0.95**
- π(B|S₁) = ε/2 = **0.05**

**S₂** (greedy = A):
- π(A|S₂) = 1 − ε + ε/2 = 0.9 + 0.05 = **0.95**
- π(B|S₂) = ε/2 = **0.05**

---

## Question 16: Multi-Step Returns and MC [4 marks]

An agent generates an episode: S₁ →(r=+1)→ S₂ →(r=+2)→ S₃ →(r=+3)→ S₄ →(r=+4)→ Terminal

**Given:** γ = 0.5

**(a)** [2 marks] Compute the MC return (full return) from S₁.

**(b)** [2 marks] Compare with the 2-step return from S₁: $G_{0:2} = r_1 + \gamma r_2 + \gamma^2 V(S_3)$ where V(S₃) = 5.5. Which is larger?

### Solution 16

**(a)** MC return from S₁:

$G_0 = 1 + 0.5(2) + 0.25(3) + 0.125(4) = 1 + 1 + 0.75 + 0.5 = \mathbf{3.25}$

**(b)** 2-step return from S₁ with V(S₃) = 5.5:

$G_{0:2} = r_1 + \gamma r_2 + \gamma^2 V(S_3) = 1 + 0.5(2) + 0.25(5.5) = 1 + 1 + 1.375 = \mathbf{3.375}$

The 2-step return (3.375) is **larger** than the MC return (3.25) because V(S₃)=5.5 is higher than the actual discounted return from S₃ onwards (3 + 0.5×4 = 5, and γ²×5 = 1.25 vs γ²×5.5 = 1.375).

---

## Question 17: Off-Policy MC — Episode Truncation [5 marks]

An agent generates an episode of length 4 under behavior policy b:

**Episode:** S₁ →(a₁)→ S₂ →(a₃)→ S₃ →(a₁)→ S₄ →(a₂)→ Terminal

Rewards: r₁=+2, r₂=+1, r₃=+3, r₄=+5

**Policy probabilities:**

| Time | Action | π(a|s) | b(a|s) |
|------|--------|--------|--------|
| t=0  | a₁     | 0.8    | 0.4    |
| t=1  | a₃     | 0.0    | 0.3    |
| t=2  | a₁     | 1.0    | 0.5    |
| t=3  | a₂     | 0.6    | 0.6    |

**(a)** [2 marks] Compute the cumulative IS ratio ρ₀:₃. What happens?

**(b)** [2 marks] What is the effective return contribution of this episode to Q(S₁, a₁) under ordinary IS?

**(c)** [1 mark] How does weighted IS handle this situation differently from ordinary IS?

### Solution 17

**(a)** Cumulative IS ratio:

$\rho_{0:3} = \frac{0.8}{0.4} \times \frac{0.0}{0.3} \times \frac{1.0}{0.5} \times \frac{0.6}{0.6} = 2.0 \times 0 \times 2.0 \times 1.0 = \mathbf{0}$

The ratio is **zero** because π(a₃|S₂) = 0 — the target policy would never take action a₃ at S₂.

**(b)** Effective contribution:

Return from S₁: G₀ = 2 + 1 + 3 + 5 = 11 (assuming γ=1 for simplicity)

Weighted return: ρ₀:₃ × G₀ = 0 × 11 = **0**

This episode contributes absolutely nothing to the Q estimate.

**(c)** Weighted IS handles this differently: episodes with ρ=0 are excluded from both numerator and denominator. In ordinary IS, ρ=0 episodes still count toward the 1/N divisor, diluting the estimate. Weighted IS avoids this dilution but introduces bias.

---

## Question 18: Blackjack MC Example [5 marks]

In a simplified blackjack game, the agent is in state (Player sum=18, Dealer showing=6, No usable ace). The agent has two actions: Hit or Stand.

After 100 episodes starting from this state:
- Hit was chosen 40 times, average return = −0.3
- Stand was chosen 60 times, average return = +0.6

**(a)** [2 marks] What are the MC estimates Q(state, Hit) and Q(state, Stand)?

**(b)** [2 marks] Under an ε-greedy policy with ε=0.1, what is π(Hit|state) and π(Stand|state)?

**(c)** [1 mark] If we now switch to greedy policy, which action is chosen? Does the agent ever explore Hit again?

### Solution 18

**(a)** MC estimates:
- Q(state, Hit) = average return from Hit = **−0.3**
- Q(state, Stand) = average return from Stand = **+0.6**

**(b)** ε-greedy with ε=0.1:
Greedy action = Stand (higher Q)
- π(Stand|state) = 1 − ε + ε/2 = 0.9 + 0.05 = **0.95**
- π(Hit|state) = ε/2 = **0.05**

**(c)** With greedy policy: **Stand** is always chosen.
The agent will never explore Hit again unless: (1) exploring starts places it in this state with Hit forced, or (2) the policy uses ε > 0. Pure greedy creates a frozen policy.

---

## Question 19: First-Visit MC with Constant-α — Effect of α [4 marks]

An agent visits state S in consecutive episodes with returns: G₁=10, G₂=2, G₃=10, G₄=2.

**Given:** Initial V(S) = 0

**(a)** [2 marks] Compute V(S) after all 4 episodes with α = 0.1.

**(b)** [2 marks] Compute V(S) after all 4 episodes with α = 0.9. Compare the two results and explain which tracks recent returns more closely.

### Solution 19

**(a)** α = 0.1:

| Episode | G | V_old | α(G − V) | V_new |
|---------|---|-------|-----------|-------|
| 1 | 10 | 0 | 0.1×10 = 1.0 | **1.0** |
| 2 | 2 | 1.0 | 0.1×1 = 0.1 | **1.1** |
| 3 | 10 | 1.1 | 0.1×8.9 = 0.89 | **1.99** |
| 4 | 2 | 1.99 | 0.1×0.01 = 0.001 | **1.991** |

**(b)** α = 0.9:

| Episode | G | V_old | α(G − V) | V_new |
|---------|---|-------|-----------|-------|
| 1 | 10 | 0 | 0.9×10 = 9.0 | **9.0** |
| 2 | 2 | 9.0 | 0.9×(−7) = −6.3 | **2.7** |
| 3 | 10 | 2.7 | 0.9×7.3 = 6.57 | **9.27** |
| 4 | 2 | 9.27 | 0.9×(−7.27) = −6.543 | **2.727** |

**Comparison:** With α=0.9, V oscillates between ~9 and ~2.7, tracking the most recent return very closely. With α=0.1, V converges slowly toward the mean (6) but barely moves. High α = recency bias, low α = long-term average.

---

## Question 20: Off-Policy MC for State Values [5 marks]

An agent generates two episodes starting from state S under behavior policy b:

![Q20 Diagram](../images/q20-offpolicy-state.svg)

**Episode 1:** S →(a₁, r=+4)→ S' →(a₂, r=+6)→ Terminal
**Episode 2:** S →(a₂, r=+8)→ Terminal

**Given:**
- γ = 0.9

**Policies at state S:**

| Action | π(a|S) | b(a|S) |
|--------|--------|--------|
| a₁     | 0.7    | 0.5    |
| a₂     | 0.3    | 0.5    |

**Policies at state S':**

| Action | π(a|S') | b(a|S') |
|--------|---------|---------|
| a₂     | 1.0     | 0.6     |

**(a)** [1 mark] Compute the return from state S for each episode.

**(b)** [2 marks] Compute the IS ratio for each episode.

**(c)** [2 marks] Compute the ordinary IS estimate and weighted IS estimate of V(S) under π.

### Solution 20

**(a)** Returns from state S:
- Episode 1: G = 4 + 0.9×6 = 4 + 5.4 = **9.4**
- Episode 2: G = 8 = **8.0**

**(b)** IS ratios:

**Episode 1:** Actions = a₁ at S, a₂ at S'
$\rho_1 = \frac{\pi(a_1|S)}{b(a_1|S)} \times \frac{\pi(a_2|S')}{b(a_2|S')} = \frac{0.7}{0.5} \times \frac{1.0}{0.6} = 1.4 \times 1.667 = \mathbf{2.333}$

**Episode 2:** Action = a₂ at S (terminal immediately)
$\rho_2 = \frac{\pi(a_2|S)}{b(a_2|S)} = \frac{0.3}{0.5} = \mathbf{0.6}$

**(c)** Estimates:

**Ordinary IS:**
$\hat{V}_{OIS}(S) = \frac{1}{2}(\rho_1 G_1 + \rho_2 G_2) = \frac{1}{2}(2.333×9.4 + 0.6×8.0)$
$= \frac{1}{2}(21.933 + 4.8) = \frac{26.733}{2} = \mathbf{13.367}$

**Weighted IS:**
$\hat{V}_{WIS}(S) = \frac{\rho_1 G_1 + \rho_2 G_2}{\rho_1 + \rho_2} = \frac{21.933 + 4.8}{2.333 + 0.6} = \frac{26.733}{2.933} = \mathbf{9.115}$

---

## Question 21: MC Control — Policy Oscillation [4 marks]

After collecting episodes, an agent has:

| State | Q(s, Left) | Q(s, Right) | Episodes(Left) | Episodes(Right) |
|-------|-----------|-------------|----------------|-----------------|
| S     | 5.0       | 5.1         | 50             | 3               |

**(a)** [2 marks] The greedy policy selects Right. With ε=0.1, what is the probability of selecting each action?

**(b)** [2 marks] Explain in 2-3 sentences why the estimate Q(S, Right)=5.1 based on only 3 episodes might be unreliable, and how this creates policy oscillation in MC control.

### Solution 21

**(a)** ε-greedy with ε=0.1, 2 actions:
Greedy action = Right (Q=5.1 > 5.0)
- π(Right|S) = 1 − ε + ε/2 = 0.9 + 0.05 = **0.95**
- π(Left|S) = ε/2 = **0.05**

**(b)** Q(S, Right) = 5.1 is based on only 3 episodes while Q(S, Left) = 5.0 is based on 50 episodes. The Right estimate has high variance (standard error ∝ 1/√3 vs 1/√50). A few lucky returns could inflate Q(Right) above Q(Left), making the greedy policy select Right, which then gets more episodes and might reveal its true (possibly lower) value. This causes the policy to flip-flop between Left and Right — **policy oscillation** — because early noisy estimates drive policy changes that alter which actions get explored.

---

## Question 22: Discounting and MC Returns — Long Episode [4 marks]

An agent generates an episode of length 6 with constant reward r=+1 at each step.

**Given:** γ = 0.5

**(a)** [2 marks] Compute the return $G_0$ from the first state.

**(b)** [2 marks] What would $G_0$ be if γ = 1.0? What would it be if the episode were infinitely long with γ = 0.5 (i.e., the geometric series sum)?

### Solution 22

**(a)** Return from first state:

$G_0 = 1 + 0.5 + 0.25 + 0.125 + 0.0625 + 0.03125 = \mathbf{1.96875}$

(Geometric sum: $\sum_{k=0}^{5} 0.5^k = \frac{1 - 0.5^6}{1 - 0.5} = \frac{1 - 0.015625}{0.5} = 1.96875$)

**(b)**
- If γ = 1.0: G₀ = 1×6 = **6.0** (undiscounted sum)
- If episode is infinite with γ = 0.5: $G_0 = \sum_{k=0}^{\infty} 0.5^k = \frac{1}{1-0.5} = \mathbf{2.0}$

Note: With γ=0.5, even an infinite episode has bounded return (2.0), and our 6-step episode already captures 1.96875/2.0 = 98.4% of the infinite-horizon value.

---

## Question 23: Off-Policy Every-Visit MC [5 marks]

An agent generates one episode where state S is visited twice:

![Q23 Diagram](../images/q23-everyvisit-offpolicy.svg)

**Episode:** S →(a₁, r=+2)→ T →(a₁, r=+1)→ S →(a₂, r=+5)→ Terminal

**Given:**
- γ = 1.0, α = 0.5
- Initial Q(S, a₁) = 0, Q(S, a₂) = 0

**Policy probabilities:**

| State | Action | π(a|s) | b(a|s) |
|-------|--------|--------|--------|
| S (t=0) | a₁  | 0.8    | 0.5    |
| T (t=1) | a₁  | 1.0    | 0.6    |
| S (t=2) | a₂  | 0.9    | 0.4    |

**(a)** [1 mark] Compute returns from each time step.

**(b)** [2 marks] Compute IS ratios and cumulative ratios for each time step.

**(c)** [2 marks] Compute the off-policy every-visit update for Q(S, a₁) at t=0 using constant-α ordinary IS.

### Solution 23

**(a)** Returns:

| Time | State-Action | Computation | G |
|------|-------------|-------------|---|
| t=2 | (S, a₂) | G₂ = 5 | **5** |
| t=1 | (T, a₁) | G₁ = 1 + 1.0×5 = 6 | **6** |
| t=0 | (S, a₁) | G₀ = 2 + 1.0×6 = 8 | **8** |

**(b)** IS ratios:

| Time | Action | π/b (single) | Cumulative ρ (from t to T−1) |
|------|--------|-------------|------------------------------|
| t=2 | a₂ at S | 0.9/0.4 = 2.25 | **2.25** |
| t=1 | a₁ at T | 1.0/0.6 = 1.667 | 1.667 × 2.25 = **3.75** |
| t=0 | a₁ at S | 0.8/0.5 = 1.6 | 1.6 × 3.75 = **6.0** |

**(c)** Off-policy update for Q(S, a₁) at t=0:

- Cumulative ρ₀:₂ = 6.0
- G₀ = 8
- ρ·G = 6.0 × 8 = 48
- Error: ρ·G − Q = 48 − 0 = 48
- α × Error = 0.5 × 48 = 24
- **Q(S, a₁) = 0 + 24 = 24**

This is very high because the cumulative IS ratio (6.0) greatly amplifies the return — target policy π favors this trajectory much more than behavior policy b does.

---

## Question 24: MC Prediction Convergence [3 marks]

An agent has run N=36 episodes from state S, obtaining returns that average to $\bar{G} = 7.5$ with sample standard deviation σ = 3.0.

**(a)** [1 mark] What is the standard error of the MC estimate?

**(b)** [2 marks] How many total episodes would be needed to reduce the standard error below 0.5?

### Solution 24

**(a)** Standard error:

$SE = \frac{\sigma}{\sqrt{N}} = \frac{3.0}{\sqrt{36}} = \frac{3.0}{6} = \mathbf{0.5}$

**(b)** Require SE < 0.5:

$\frac{3.0}{\sqrt{N}} < 0.5$

$\sqrt{N} > 6.0$

$N > 36$

Need at least **N = 37 episodes** to achieve SE < 0.5.

---

## Question 25: Comparing On-Policy and Off-Policy for Same Episode [5 marks]

An agent generates one episode under behavior policy b:

![Q25 Diagram](../images/q25-on-vs-off.svg)

**Episode:** P →(a₁, r=+3)→ Q →(a₁, r=+9)→ Terminal

**Given:**
- γ = 0.95, α = 0.3
- Initial Q(P, a₁) = 4, Q(Q, a₁) = 2
- π(a₁|P) = 1.0, b(a₁|P) = 0.4
- π(a₁|Q) = 0.8, b(a₁|Q) = 0.5

**(a)** [1 mark] Compute returns from each state.

**(b)** [2 marks] Compute on-policy MC updates (constant-α, no IS).

**(c)** [2 marks] Compute off-policy MC updates (ordinary IS, constant-α). Show how the IS ratio amplifies or dampens the update.

### Solution 25

**(a)** Returns:
- G₁ (from Q) = r₂ = 9 → **G₁ = 9**
- G₀ (from P) = r₁ + γ·G₁ = 3 + 0.95×9 = 3 + 8.55 = **G₀ = 11.55**

**(b)** On-policy MC updates (no IS):

| State | Q_old | G | Error | α×Error | Q_new |
|-------|-------|---|-------|---------|-------|
| Q(Q, a₁) | 2 | 9 | 9−2=7 | 0.3×7=2.1 | **4.1** |
| Q(P, a₁) | 4 | 11.55 | 11.55−4=7.55 | 0.3×7.55=2.265 | **6.265** |

**(c)** Off-policy MC updates (ordinary IS):

IS ratios:
- ρ₁:₁ (for Q) = π(a₁|Q)/b(a₁|Q) = 0.8/0.5 = **1.6**
- ρ₀:₁ (for P) = [π(a₁|P)/b(a₁|P)] × ρ₁:₁ = (1.0/0.4) × 1.6 = 2.5 × 1.6 = **4.0**

| State | Q_old | G | ρ | ρ·G | Error (ρ·G−Q) | α×Error | Q_new |
|-------|-------|---|---|-----|----------------|---------|-------|
| Q(Q, a₁) | 2 | 9 | 1.6 | 14.4 | 14.4−2=12.4 | 0.3×12.4=3.72 | **5.72** |
| Q(P, a₁) | 4 | 11.55 | 4.0 | 46.2 | 46.2−4=42.2 | 0.3×42.2=12.66 | **16.66** |

**How IS amplifies:** The ρ=4.0 for P means target policy π would take this trajectory 4× more often than behavior b, so the return is amplified accordingly. Q(P, a₁) jumps from 4 to 16.66 (vs 6.265 on-policy).

---

## Question 26: Weighted IS — Incremental Implementation [5 marks]

An agent uses incremental weighted IS to update Q(S, a). Episodes arrive one at a time:

| Episode | Return G | ρ    |
|---------|----------|------|
| 1       | 8        | 2.0  |
| 2       | 5        | 0.5  |
| 3       | 10       | 1.2  |

The incremental weighted IS update formula is:
$$Q_{n+1} = Q_n + \frac{\rho_n}{C_n}[G_n - Q_n]$$
where $C_n = C_{n-1} + \rho_n$ (cumulative sum of weights).

**Given:** Initial Q = 0, C₀ = 0

**(a)** [2 marks] Compute Q after episode 1.

**(b)** [2 marks] Compute Q after episode 2.

**(c)** [1 mark] Compute Q after episode 3. Verify this equals the batch weighted IS formula.

### Solution 26

**(a)** After Episode 1 (G=8, ρ=2.0):

- C₁ = C₀ + ρ₁ = 0 + 2.0 = **2.0**
- Q₁ = Q₀ + (ρ₁/C₁)(G₁ − Q₀) = 0 + (2.0/2.0)(8 − 0) = 1.0 × 8 = **8.0**

**(b)** After Episode 2 (G=5, ρ=0.5):

- C₂ = C₁ + ρ₂ = 2.0 + 0.5 = **2.5**
- Q₂ = Q₁ + (ρ₂/C₂)(G₂ − Q₁) = 8.0 + (0.5/2.5)(5 − 8) = 8.0 + 0.2×(−3) = 8.0 − 0.6 = **7.4**

**(c)** After Episode 3 (G=10, ρ=1.2):

- C₃ = C₂ + ρ₃ = 2.5 + 1.2 = **3.7**
- Q₃ = Q₂ + (ρ₃/C₃)(G₃ − Q₂) = 7.4 + (1.2/3.7)(10 − 7.4) = 7.4 + 0.324×2.6 = 7.4 + 0.843 = **8.243**

**Verification with batch formula:**
$Q = \frac{\sum \rho_i G_i}{\sum \rho_i} = \frac{2.0×8 + 0.5×5 + 1.2×10}{2.0 + 0.5 + 1.2} = \frac{16 + 2.5 + 12}{3.7} = \frac{30.5}{3.7} = \mathbf{8.243}$ ✓

---

## Question 27: MC with Function Approximation Intuition [3 marks]

In tabular MC, V(S) is updated only when S is visited. Consider a gridworld with 100 states where an episode of length 10 visits 10 distinct states.

**(a)** [1 mark] How many state values are updated after this single episode?

**(b)** [2 marks] If we run 50 such episodes (each visiting 10 random states), what is the expected number of visits per state? What does this mean for convergence speed of rarely-visited states?

### Solution 27

**(a)** Only the **10 visited states** have their V(s) updated. The other 90 states remain unchanged.

**(b)** Expected visits per state = (50 episodes × 10 states/episode) / 100 states = **5 visits per state on average**.

However, this is the average — some states may be visited 0 times while others might be visited 15+ times. States that are rarely reachable under the policy (corner states, dead ends) will have very few visits and thus high-variance, unreliable value estimates. MC convergence is only guaranteed as each state's visit count → ∞, and rarely-visited states converge much slower.

---

## Question 28: Off-Policy MC — High Variance Example [5 marks]

An agent generates 4 episodes from state S (action a) under behavior policy b:

| Episode | Return G | ρ (cumulative) |
|---------|----------|----------------|
| 1       | 5        | 0.1            |
| 2       | 3        | 8.0            |
| 3       | 7        | 0.3            |
| 4       | 4        | 6.0            |

**(a)** [2 marks] Compute the ordinary IS estimate.

**(b)** [2 marks] Compute the weighted IS estimate.

**(c)** [1 mark] Explain why the ordinary IS estimate has much higher variance in this case.

### Solution 28

**(a)** Ordinary IS estimate:

$\hat{V}_{OIS} = \frac{1}{4}\sum_i \rho_i G_i = \frac{1}{4}(0.1×5 + 8.0×3 + 0.3×7 + 6.0×4)$

$= \frac{1}{4}(0.5 + 24 + 2.1 + 24) = \frac{50.6}{4} = \mathbf{12.65}$

**(b)** Weighted IS estimate:

$\hat{V}_{WIS} = \frac{\sum_i \rho_i G_i}{\sum_i \rho_i} = \frac{50.6}{0.1 + 8.0 + 0.3 + 6.0} = \frac{50.6}{14.4} = \mathbf{3.514}$

**(c)** The ordinary IS estimate (12.65) has very high variance because extreme ratios (ρ=8.0, ρ=6.0) heavily weight certain episodes. Episode 2 contributes 24 and Episode 4 contributes 24 despite modest returns (3 and 4), because their high ρ values amplify the contribution. Weighted IS normalizes by total weight, preventing high-ρ episodes from dominating.

---

## Question 29: MC for Action-Value Estimation — Multiple Actions [5 marks]

An agent in state S has 3 actions. After many episodes:

| (State, Action) | Visits | Returns observed | Average return |
|-----------------|--------|-----------------|----------------|
| (S, a₁)        | 20     | sum = 140       | 7.0            |
| (S, a₂)        | 5      | sum = 45        | 9.0            |
| (S, a₃)        | 30     | sum = 180       | 6.0            |

**(a)** [2 marks] Under a greedy policy, which action is selected? Under ε-greedy with ε=0.15?

**(b)** [2 marks] The estimate for a₂ is based on only 5 visits. Compute the standard error if the sample standard deviation of the 5 returns for a₂ is 4.0.

**(c)** [1 mark] Explain the exploration-exploitation dilemma this illustrates.

### Solution 29

**(a)** Greedy policy: argmax Q → **a₂** (Q=9.0 is highest)

ε-greedy with ε=0.15, 3 actions:
- π(a₂|S) = 1 − ε + ε/3 = 0.85 + 0.05 = **0.9**
- π(a₁|S) = ε/3 = 0.15/3 = **0.05**
- π(a₃|S) = ε/3 = 0.15/3 = **0.05**

**(b)** Standard error for a₂:

$SE = \frac{\sigma}{\sqrt{n}} = \frac{4.0}{\sqrt{5}} = \frac{4.0}{2.236} = \mathbf{1.789}$

This is quite large — the 95% confidence interval is roughly 9.0 ± 3.58, meaning the true value could be anywhere from 5.4 to 12.6.

**(c)** **Exploration-exploitation dilemma:** The greedy policy selects a₂ because its average return (9.0) is highest, but this estimate is based on only 5 episodes with high uncertainty (SE=1.79). The "worse" action a₁ (Q=7.0) has 20 visits and is much more reliably estimated. The agent might be exploiting a noisy high estimate while the true best action could be a₁.

---

## Question 30: Batch MC vs Online MC [4 marks]

An agent processes 4 returns for state S: G₁=10, G₂=6, G₃=8, G₄=12.

**(a)** [2 marks] Compute the batch estimate (simple average after all 4).

**(b)** [2 marks] Compute the incremental estimates V₁, V₂, V₃, V₄ using the incremental formula $V_n = V_{n-1} + \frac{1}{n}(G_n - V_{n-1})$, starting from V₀=0. Verify V₄ matches the batch estimate.

### Solution 30

**(a)** Batch estimate:
$(10 + 6 + 8 + 12)/4 = 36/4 = \mathbf{9.0}$

**(b)** Incremental estimates:

| n | G_n | V_{n-1} | 1/n | G_n − V_{n-1} | (1/n)(G_n−V) | V_n |
|---|-----|---------|-----|----------------|--------------|-----|
| 1 | 10 | 0 | 1.0 | 10 | 10.0 | **10.0** |
| 2 | 6 | 10.0 | 0.5 | −4 | −2.0 | **8.0** |
| 3 | 8 | 8.0 | 0.333 | 0 | 0 | **8.0** |
| 4 | 12 | 8.0 | 0.25 | 4 | 1.0 | **9.0** |

V₄ = **9.0** = batch estimate ✓

---

## Question 31: Off-Policy MC — Behavior Policy Design [4 marks]

A target policy π is deterministic: π(Left|S)=1.0. You need to design a behavior policy b for off-policy MC.

**(a)** [2 marks] If b(Left|S) = 0.9, what is the IS ratio for each episode that takes Left at S? If b(Left|S) = 0.5?

**(b)** [2 marks] Which behavior policy will produce lower variance in the IS estimate? Explain intuitively using the concept of coverage.

### Solution 31

**(a)** IS ratios:

- If b(Left|S) = 0.9: ρ = π/b = 1.0/0.9 = **1.111**
- If b(Left|S) = 0.5: ρ = π/b = 1.0/0.5 = **2.0**

**(b)** **b(Left|S) = 0.9 produces lower variance.**

When the behavior policy is close to the target policy (b ≈ π), the IS ratio is close to 1, meaning each episode's return is barely scaled. With b=0.9, the ratio is only 1.111, so weighted returns are almost the same as raw returns — low variance.

With b=0.5, the ratio is 2.0 — returns are doubled, and episodes taking Right (which π never would) contribute 0. This creates high variance: some episodes contribute 2× their return while others contribute nothing.

**General principle:** The closer b is to π while maintaining coverage (b(a|s)>0 whenever π(a|s)>0), the lower the IS variance. However, b must maintain sufficient exploration to visit diverse trajectories — this is the bias-variance tradeoff in behavior policy design.

---

*End of Solutions*
