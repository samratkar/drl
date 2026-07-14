---
layout: post
---

# Numerical Questions (Lecture 10: PPO)

**Question 1: PPO Objective Calculation**
Assume $\epsilon = 0.2$. An agent gathers data using an old policy $\pi_{old}$. For a specific state $S$ and action $A$:
* The Advantage was calculated as $A_t = +10$.
* The probability of taking action A under the old policy was $\pi_{old}(A|S) = 0.4$.

We are now doing a PPO gradient update. Our *current* online neural network evaluates the state, and its new updated probability for action A is $\pi_{new}(A|S) = 0.6$.

1. Calculate the probability ratio $r_t(\theta)$.
2. Calculate the unclipped objective value.
3. Calculate the clipped objective value.
4. What is the final value of the PPO Surrogate Objective $L^{CLIP}$ for this action? Is the gradient dead or alive?
