---
layout: post
---

# Numerical Question Solutions - Lecture 4

1. **MC Return Calculation:**
   $G_0 = R_1 + \gamma R_2 + \gamma^2 R_3$
   $G_0 = 2 + 0.9(-1) + 0.81(4) = 2 - 0.9 + 3.24 = 4.34$

2. **TD(0) Update:**
   Target = $R_{t+1} + \gamma V(S_{t+1}) = 2.0 + 0.9(5.0) = 2.0 + 4.5 = 6.5$
   Error $\delta = 6.5 - 10.0 = -3.5$
   $V(S_t) \leftarrow 10.0 + 0.1(-3.5) = 10.0 - 0.35 = 9.65$

3. **Importance Sampling Ratio:**
   $\rho = \frac{\pi(Hit)}{\beta(Hit)} \cdot \frac{\pi(Hit)}{\beta(Hit)} \cdot \frac{\pi(Stick)}{\beta(Stick)}$
   $\rho = \frac{0.8}{0.5} \cdot \frac{0.8}{0.5} \cdot \frac{0.2}{0.5} = 1.6 \cdot 1.6 \cdot 0.4 = 2.56 \cdot 0.4 = 1.024$

4. **Sarsa vs. Q-learning Target:**
   a) **Sarsa:** Target = $R + \gamma Q(S', a_{selected})$
      Since action $a_1$ was selected: $Target = 5 + 1.0(10) = 15$
   b) **Q-learning:** Target = $R + \gamma \max_a Q(S', a)$
      $Target = 5 + 1.0(\max(10, 2, 6)) = 5 + 10 = 15$
      (Note: here they match because $a_1$ happened to be the greedy action).

5. **Weighted Importance Sampling:**
   $V = \frac{\rho_1 G_1 + \rho_2 G_2}{\rho_1 + \rho_2}$
   $V = \frac{1.5(10) + 0.5(5)}{1.5 + 0.5} = \frac{15 + 2.5}{2.0} = \frac{17.5}{2.0} = 8.75$
