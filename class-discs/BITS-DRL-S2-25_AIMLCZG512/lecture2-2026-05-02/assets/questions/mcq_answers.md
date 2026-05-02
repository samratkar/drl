# Chapter 2: Multi-Armed Bandits — MCQ Answer Key

*Based on Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.), Chapter 2*

---

### Q1. Evaluative vs Instructive Feedback
**Answer**: (b)

Evaluative feedback indicates how good the taken action was, but not whether it was the best or worst. This is what creates the need for active exploration in RL.

---

### Q2. k-Armed Bandit Definition
**Answer**: (b)

The agent tries to maximize the total expected reward over a time period.

---

### Q3. True Value vs Estimated Value
**Answer**: (b)

$Q_t(a)$ is the agent's estimate of $q_*(a)$ at time $t$, based on observed rewards.

---

### Q4. Greedy Action
**Answer**: (c)

Always selecting the greedy action may lock onto a suboptimal arm and never discover better alternatives.

---

### Q5. Epsilon-Greedy Exploration
**Answer**: (b)

With probability $\epsilon = 0.1$ a random arm is chosen uniformly from all 10, so each arm gets $0.1/10 = 0.01$. (Note: the greedy arm also has this $0.01$ chance from the explore branch, plus the $0.9$ from the exploit branch.)

---

### Q6. When to Increase Exploration
**Answer**: (b)

$\epsilon$ should be larger when the rewards have high variance or $q_*(a)$ changes over time.

---

### Q7. Sample-Average Method
**Answer**: (b)

It converges to $q_*(a)$ by the law of large numbers.

---

### Q8. Incremental Update Rule
**Answer**: (c)

The incremental update $Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]$ computes the simple average of all $n$ observed rewards for that arm.

---

### Q9. Constant Step-Size Weighting
**Answer**: (b)

The most recent reward $R_n$ has weight $\alpha(1-\alpha)^0 = \alpha$, which is the largest.

---

### Q10. Stationary vs Non-Stationary
**Answer**: (b)

The sample-average method gives equal weight to all past rewards, including stale ones from a distribution that no longer applies.

---

### Q11. Convergence Conditions
**Answer**: (b)

$\alpha_n = 1/n$ satisfies both: $\sum 1/n = \infty$ (harmonic series) and $\sum 1/n^2 = \pi^2/6 < \infty$.

---

### Q12. Why Constant $\alpha$ Doesn't Converge
**Answer**: (b)

The estimate never fully settles, allowing it to continuously track a changing reward distribution.

---

### Q13. Initial Bias — Sample Average
**Answer**: (c)

$Q_2 = Q_1 + \frac{1}{1}(R_1 - Q_1) = 5 + 1 \times (1 - 5) = 1.0 = R_1$. The initial value is completely replaced after one step.

---

### Q14. Initial Bias — Constant Alpha
**Answer**: (b)

$Q_1$ still contributes $5 \times 0.35 = 1.74$ to the current estimate after 10 steps.

---

### Q15. Optimistic Initial Values — Mechanism
**Answer**: (b)

Every arm's initial estimate is above the true value, so every reward feels "disappointing," pushing the agent to try other arms.

---

### Q16. UCB — No Random Branch
**Answer**: (b)

UCB uses a single argmax over a combined score (value + uncertainty bonus), with no random selection.

---

### Q17. UCB — Under-Explored Arms
**Answer**: (c)

The bonus $c\sqrt{\ln t / N_t(a)} \to \infty$ when $N_t(a) = 0$, guaranteeing the arm is tried first.

---

### Q18. Gradient Bandit — Preferences
**Answer**: (c)

Nothing changes — only relative differences between preferences matter. Softmax is shift-invariant: $\frac{e^{H+c}}{\sum e^{H+c}} = \frac{e^c \cdot e^H}{e^c \cdot \sum e^H} = \frac{e^H}{\sum e^H}$.

---

### Q19. Gradient Bandit — Baseline Purpose
**Answer**: (b)

The baseline doesn't affect the expected gradient direction, but dramatically reduces variance by reframing the question from "was this reward positive?" to "was this reward better than average?"

---

### Q20. Parameter Study — Overall Lesson
**Answer**: (b)

Too little exploration (left) misses the best arm; too much exploration (right) wastes time on bad arms; the peak is the sweet spot.
