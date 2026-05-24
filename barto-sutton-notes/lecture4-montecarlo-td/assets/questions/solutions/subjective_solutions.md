---
layout: post
---

# Subjective Question Solutions - Lecture 4

1. **MC vs. TD:** Discuss the fundamental differences between Monte Carlo and Temporal-Difference methods. Consider aspects like bootstrapping, update frequency, and bias-variance trade-offs.

   - **MC:** waits for episode end, unbiased, high variance, no bootstrapping.
   - **TD:** updates every step, biased (initially), lower variance, bootstraps.
   - TD can learn from incomplete episodes and continuing tasks; MC requires termination.

2. **First-visit vs. Every-visit MC:** Define both methods. Why is First-visit MC considered an unbiased estimate of the value function?

   - **First-visit:** Counts only the first time a state $s$ is visited in an episode.
   - **Every-visit:** Counts every time $s$ is visited.
   - First-visit is unbiased because each return $G$ is an independent sample. Every-visit is slightly biased but converges faster.

3. **Importance Sampling Intuition:** Explain why importance sampling is necessary for off-policy learning. What is the difference between Ordinary and Weighted importance sampling in terms of bias and variance?

   - IS is needed because we are calculating the expected value under $\pi$ using samples from $b$.
   - **Ordinary IS:** Unbiased but can have extremely high (even infinite) variance.
   - **Weighted IS:** Biased (converges to 0) but much lower variance and always finite.

4. **The Advantage of TD(0):** Why is TD(0) often more efficient than MC prediction for tasks that satisfy the Markov property? Mention the concept of "Certainty-Equivalence."

   - TD(0) converges to the **certainty-equivalence** estimate. It makes full use of the Markov property.
   - MC ignores the Markov property and treats every sequence as independent, which is less efficient if the state transitions are indeed Markovian.

5. **Sarsa vs. Q-learning:** Using the Cliff Walking task as an example, explain why these two algorithms learn different policies even though they are both trying to maximize reward.

   - **Sarsa:** On-policy. It evaluates the current exploratory policy. It "sees" the $-100$ reward from falling off the cliff occasionally due to $\epsilon$ and learns a safer path.
   - **Q-learning:** Off-policy. It evaluates the greedy policy. It assumes it will act optimally next, so it tries to learn the path right along the cliff edge (which is risky during exploration).

6. **The Maximization Bias Trap:** Describe a simple scenario where Q-learning would exhibit maximization bias. How does Double Q-learning fix this?

   - **Scenario:** A state has many actions with true value $0$, but noisy estimates. $\max$ will pick the one with the highest positive noise, leading to a large positive bias.
   - **Double Q-learning:** Decouples selection ($argmax$) from evaluation. It uses $Q_1$ to pick the best action and $Q_2$ to tell you its value. Noise in $Q_1$ is unlikely to match noise in $Q_2$ at the same spot.

7. **Expected Sarsa's Robustness:** Explain why Expected Sarsa is generally more robust to changes in the step-size $\alpha$ and exploration parameter $\epsilon$ compared to standard Sarsa.

   - Standard Sarsa uses a sample action $A_{t+1}$. If $\alpha$ is high, a single unlucky "exploratory" action can jump the $Q$-value significantly.
   - Expected Sarsa uses the probability-weighted average, so exploratory actions are smoothed out in the update.

8. **Incremental Updates:** Derive the incremental update rule for the sample average of a sequence of rewards. How does this rule generalize to include a constant step-size $\alpha$?

   - $V_n = \frac{R_1 + \dots + R_{n-1}}{n-1}$
   - $V_{n+1} = V_n + \frac{1}{n} [R_n - V_n]$
   - Replacing $1/n$ with constant $\alpha$ leads to an exponential recency-weighted average.

9. **Infinite Variance in IS:** Explain Example 5.5 from the book. How can a simple MDP lead to ordinary importance sampling having infinite variance?

   - If the ratio $\rho$ is frequently large, the variance of $\rho G$ (which involves $\rho^2$) can diverge.
   - In Example 5.5, the product of ratios in a loop leads to an expected value of the squared ratio that is a geometric series with ratio $> 1$.

10. **Afterstates:** What are "afterstates" and why can learning values for them be more efficient in games like Tic-Tac-Toe or Chess?

    - Different state-action pairs can lead to the same configuration (e.g., placing a cross in the center).
    - By learning the value of the *result* (afterstate), we generalize learning across all actions that produce that result.
