---
layout: post
---

# Subjective Question Solutions - Lecture 2

1. **Exploration-Exploitation Trade-off:**
   - **Exploitation** maximizes short-term reward by choosing the current best.
   - **Exploration** improves long-term reward by gathering info about other arms.
   - You can't do both in one step. Too much exploitation leads to getting stuck at local optima; too much exploration wastes time on bad actions.

2. **Sample-Average vs. Constant Step-size:**
   - Sample-average $Q_{n+1} = \frac{1}{n} \sum R_i$ gives equal weight to all data.
   - Constant $\alpha$ gives weight $\alpha(1-\alpha)^{n-i}$ to $R_i$.
   - For non-stationary problems, old data is wrong. Constant $\alpha$ causes exponential decay of weights for old data, allowing the agent to "forget" the old distribution.

3. **Optimistic Initial Values:**
   - Setting $Q_1$ high makes every real reward look like a "loss" initially. This forces the greedy algorithm to cycle through all actions to see if any match the high expectation.
   - Limitation: It only works at the start. It doesn't help with non-stationary changes later in the run.

4. **UCB Action Selection:**
   - UCB adds $\text{bonus} = c \sqrt{\ln t / N_t(a)}$ to the estimate $Q_t(a)$.
   - Small $N_t(a)$ (little data) makes the bonus large. Large $t$ (time passed) slowly increases the bonus for actions not chosen recently.
   - It picks the action that has either a high value or a high uncertainty.

5. **Gradient Bandit Algorithm:**
   - It learns preferences $H_t(a)$ and uses softmax for action selection.
   - The update is $H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R}_t)(\dots)$.
   - The baseline $\bar{R}_t$ (average reward) makes the algorithm compare $R_t$ to "what I usually get." This reduces the variance of the updates significantly.

6. **The 10-Armed Testbed:**
   - $\epsilon=0$ (Greedy) stops exploring once it thinks it found the best. Due to noise, it often locks onto a mediocre arm.
   - $\epsilon=0.1$ continues to try other arms, eventually identifying the true best arm, thus reaching higher average reward in the long run.

7. **Convergence Conditions:**
   - 1) $\sum \alpha_n = \infty$: steps are large enough to overcome any start.
   - 2) $\sum \alpha_n^2 < \infty$: steps shrink small enough to settle.
   - Constant $\alpha$ violates #2 because $\sum \alpha^2 = \infty$. This is intentional so the agent can keep tracking a changing environment forever.

8. **Non-stationarity Example:**
   - A news recommendation system where trends change hourly. 
   - Sample-average would remember what was popular last year just as much as what is popular today, making it unresponsive to current trends.
