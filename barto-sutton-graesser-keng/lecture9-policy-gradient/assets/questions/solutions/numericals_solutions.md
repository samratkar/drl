---
layout: post
---

# Solutions to Numerical Questions (Lecture 9)

**Question 1 Solution:**
1. **Advantage Calculation:**
   Advantage = $G_t - V(S)$
   Advantage = $10 - 4 = +6$

2. **Probability Shift:**
   The Advantage is positive (+6). This means taking action A resulted in a return that was *better* than the average return usually expected from state S. 
   Therefore, the gradient ascent step $\theta = \theta + \alpha (Advantage) \nabla \ln \pi(A|S)$ will push the weights in a direction that **increases** the probability of taking Action A in the future. (Consequently, because probabilities must sum to 1, the probabilities of B and C will slightly decrease).
