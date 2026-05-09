---
layout: post
---

# Subjective Questions - Lecture 4 (Monte Carlo & TD Learning)

1. **MC vs. TD:** Discuss the fundamental differences between Monte Carlo and Temporal-Difference methods. Consider aspects like bootstrapping, update frequency, and bias-variance trade-offs.

2. **First-visit vs. Every-visit MC:** Define both methods. Why is First-visit MC considered an unbiased estimate of the value function?

3. **Importance Sampling Intuition:** Explain why importance sampling is necessary for off-policy learning. What is the difference between Ordinary and Weighted importance sampling in terms of bias and variance?

4. **The Advantage of TD(0):** Why is TD(0) often more efficient than MC prediction for tasks that satisfy the Markov property? Mention the concept of "Certainty-Equivalence."

5. **Sarsa vs. Q-learning:** Using the Cliff Walking task as an example, explain why these two algorithms learn different policies even though they are both trying to maximize reward.

6. **The Maximization Bias Trap:** Describe a simple scenario where Q-learning would exhibit maximization bias. How does Double Q-learning fix this?

7. **Expected Sarsa's Robustness:** Explain why Expected Sarsa is generally more robust to changes in the step-size $\alpha$ and exploration parameter $\epsilon$ compared to standard Sarsa.

8. **Incremental Updates:** Derive the incremental update rule for the sample average of a sequence of rewards. How does this rule generalize to include a constant step-size $\alpha$?

9. **Infinite Variance in IS:** Explain Example 5.5 from the book. How can a simple MDP lead to ordinary importance sampling having infinite variance?

10. **Afterstates:** What are "afterstates" and why can learning values for them be more efficient in games like Tic-Tac-Toe or Chess?
