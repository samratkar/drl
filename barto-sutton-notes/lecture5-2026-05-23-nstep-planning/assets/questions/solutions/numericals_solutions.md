# Numerical Question Solutions - Lecture 5

1. **Calculating n-step Returns:**
   Given $R_1=2, R_2=-1, R_3=4, R_4=10$ and $V(S_4)=5.0, \gamma=0.9$.
   
   a) **1-step return $G_{0:1}$:**
   $G_{0:1} = R_1 + \gamma V(S_1)$ -> Wait, we don't have $V(S_1)$.
   Actually, the question implies we are at $t=0$ and calculating returns at different horizons.
   Usually, $G_{0:n}$ uses $V(S_n)$. 
   If we only have $V(S_4)$, we can only calculate the 4-step return accurately if $V(S_4)$ is the only bootstrap point.
   
   Let's assume the question meant we have the values for all states, but for simplicity, let's look at the formula:
   $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$.
   
   Assuming we have $V(S_1)=V(S_2)=V(S_3)=V(S_4)=5.0$ for this calculation:
   a) $G_{0:1} = R_1 + 0.9(5.0) = 2 + 4.5 = 6.5$
   b) $G_{0:2} = R_1 + 0.9 R_2 + 0.9^2 V(S_2) = 2 + 0.9(-1) + 0.81(5.0) = 2 - 0.9 + 4.05 = 5.15$
   c) $G_{0:3} = R_1 + 0.9 R_2 + 0.81 R_3 + 0.9^3 V(S_3) = 2 - 0.9 + 0.81(4) + 0.729(5.0) = 1.1 + 3.24 + 3.645 = 7.985$
   d) $G_{0:4} = R_1 + 0.9 R_2 + 0.81 R_3 + 0.729 R_4 + 0.9^4 V(S_4) = 2 - 0.9 + 3.24 + 0.729(10) + 0.6561(5.0) = 4.34 + 7.29 + 3.2805 = 14.9105$

2. **Importance Sampling Ratio:**
   $\rho_{0:2} = \frac{\pi(A_0|S_0)}{b(A_0|S_0)} \cdot \frac{\pi(A_1|S_1)}{b(A_1|S_1)} \cdot \frac{\pi(A_2|S_2)}{b(A_2|S_2)}$
   $\rho_{0:2} = \frac{0.8}{0.5} \cdot \frac{0.8}{0.5} \cdot \frac{0.2}{0.5} = 1.6 \cdot 1.6 \cdot 0.4 = 2.56 \cdot 0.4 = 1.024$.

3. **Dyna-Q+ Exploration Bonus:**
   Bonus = $\kappa \sqrt{\tau} = 0.01 \cdot \sqrt{1000} \approx 0.01 \cdot 31.62 = 0.3162$.
   The simulated reward used in planning becomes $r + 0.3162$.

4. **Tree Backup Returns:**
   $G_{t:t+2} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q(S_{t+1}, a) + \gamma \pi(A_{t+1}|S_{t+1}) [R_{t+2} + \gamma V(S_{t+2})]$
   Given $\gamma=1.0$:
   $G_{t:t+2} = 2 + (0.4 \cdot 10) + 0.6 \cdot [3 + 1.0 \cdot 5]$
   $G_{t:t+2} = 2 + 4 + 0.6 \cdot [8] = 6 + 4.8 = 10.8$.

5. **Wait-and-See return:**
   Since the episode ends at $T=3$, the 3-step return $G_{0:3}$ is just the full return (Monte Carlo return) because there is no $S_3$ to bootstrap from (or $V(S_3)=0$).
   $G_{0:3} = R_1 + \gamma R_2 + \gamma^2 R_3 = 1 + 0.9(1) + 0.81(1) = 1 + 0.9 + 0.81 = 2.71$.
   It does **not** involve bootstrapping.
