---
layout: post
---

# Solutions to Subjective Questions (Lecture 5)

1. **Bootstrapping vs. Sampling:** Bootstrapping is updating an estimate based on other learned estimates, rather than waiting for the final true outcome. 
   - **DP** bootstraps (updates states based on successor states) but does not sample (it uses the true transition model).
   - **MC** samples (learns from actual experience) but does not bootstrap (waits for the full return $G_t$).
   - **TD** methods combine both: they learn from sampled experience and bootstrap their updates based on the current estimate of the next state's value.

2. **On-Policy vs. Off-Policy:** 
   - **SARSA (On-Policy):** Learns the value of the policy it is currently executing. In cliff-walking, because the $\epsilon$-greedy policy occasionally takes random actions, walking right next to the cliff is dangerous (a random action might cause a fall). SARSA learns to take a safer path further from the cliff.
   - **Q-learning (Off-Policy):** Learns the value of the *optimal* policy, regardless of the agent's exploratory actions. It learns the path directly on the edge of the cliff because it evaluates the $\max_a Q(S', a)$ (assuming it will always act perfectly), even if its actual $\epsilon$-greedy exploration occasionally causes it to fall off.

3. **Maximization Bias:** Q-learning uses $\max_a Q(s', a)$ to estimate the maximum expected value. However, because $Q$-values are noisy estimates, the maximum of the estimates is strictly greater than or equal to the true maximum value. This leads to an overestimation of action values (maximization bias). Double Q-learning solves this by maintaining two separate Q-tables ($Q_1$ and $Q_2$). It uses one table to *select* the maximizing action, and the other table to *evaluate* its value: $Q_2(s', 	ext{argmax}_a Q_1(s', a))$. This decoupling provides an unbiased estimate.
