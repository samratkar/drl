# Solutions to Numerical Questions (Lecture 10)

**Question 1 Solution:**
1. **Probability Ratio:**
   $r_t(\theta) = \frac{\pi_{new}}{\pi_{old}} = \frac{0.6}{0.4} = 1.5$

2. **Unclipped Objective:**
   $r_t(\theta) \times A_t = 1.5 \times 10 = +15$

3. **Clipped Objective:**
   The clipping threshold is $1 + \epsilon = 1.2$.
   $\text{clip}(1.5, 0.8, 1.2) = 1.2$
   Clipped Objective = $1.2 \times 10 = +12$

4. **Final PPO Objective:**
   $L^{CLIP} = \min(15, 12) = +12$
   Because the objective hit the flat clipping ceiling ($+12$), the derivative (gradient) of this flat line with respect to $\theta$ is **0**. The gradient is **dead**. PPO successfully stopped the network from increasing the probability of this action any further during this epoch, protecting the policy from a destructive update!
