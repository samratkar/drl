---
layout: post
---

# Solutions to Numerical Questions (Chapter 9)

**Question 1 Solution:**
1. **Estimate of current state:** $\hat{v}(S_t, \mathbf{w}_t) = \mathbf{w}_t^T \mathbf{x}(S_t) = (0.5)(1) + (-0.1)(2) + (1.0)(0) = 0.5 - 0.2 = 0.3$
2. **Estimate of next state:** $\hat{v}(S_{t+1}, \mathbf{w}_t) = \mathbf{w}_t^T \mathbf{x}(S_{t+1}) = (0.5)(1) + (-0.1)(0) + (1.0)(1) = 0.5 + 1.0 = 1.5$
3. **TD Error:** $\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$
   $\delta_t = 10 + 0.9(1.5) - 0.3 = 10 + 1.35 - 0.3 = 11.05$
4. **New Weight Vector:** $\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \mathbf{x}(S_t)$
   $\mathbf{w}_{t+1} = [0.5, -0.1, 1.0]^T + 0.1(11.05)[1, 2, 0]^T$
   $\mathbf{w}_{t+1} = [0.5, -0.1, 1.0]^T + [1.105, 2.21, 0]^T$
   $\mathbf{w}_{t+1} = [1.605, 2.11, 1.0]^T$

**Question 2 Solution:**
The rule of thumb for tile coding step-size is $\alpha = \frac{1}{m \cdot c}$, where $m$ is the number of tilings (the number of active features per state) and $c$ is the inverse of the desired step fraction.
We want to move 1/4 of the way, so $c = 4$.
We have 8 tilings, so $m = 8$.
$\alpha = \frac{1}{8 \cdot 4} = \frac{1}{32} = 0.03125$
