---
layout: post
---

# Numerical Question Solutions - Lecture 4

1. **MC Return Calculation:** Given an episode: $S_0, A_0, R_1=2, S_1, A_1, R_2=-1, S_2, A_2, R_3=4, Terminal$. Assume $\gamma = 0.9$. Calculate the return $G_0$ observed from state $S_0$.

   $G_0 = R_1 + \gamma R_2 + \gamma^2 R_3$
   $G_0 = 2 + 0.9(-1) + 0.81(4) = 2 - 0.9 + 3.24 = 4.34$

2. **TD(0) Update:** Current estimate $V(S_t) = 10.0$. The agent takes an action, receives reward $R_{t+1} = 2.0$ and transitions to $S_{t+1}$. The estimate for the next state is $V(S_{t+1}) = 5.0$. Using $\alpha = 0.1$ and $\gamma = 0.9$, calculate the updated value $V(S_t)$.

   Target = $R_{t+1} + \gamma V(S_{t+1}) = 2.0 + 0.9(5.0) = 2.0 + 4.5 = 6.5$
   Error $\delta = 6.5 - 10.0 = -3.5$
   $V(S_t) \leftarrow 10.0 + 0.1(-3.5) = 10.0 - 0.35 = 9.65$

3. **Importance Sampling Ratio:** Target policy $\pi(a \mid s) = 0.8$ for action 'Hit', $0.2$ for 'Stick'. Behavior policy $b(a \mid s) = 0.5$ for action 'Hit', $0.5$ for 'Stick'. An episode follows the sequence: $Hit, Hit, Stick$. Calculate the importance sampling ratio $\rho_{0:2}$ for this episode.

   $\rho = \frac{\pi(Hit)}{\beta(Hit)} \cdot \frac{\pi(Hit)}{\beta(Hit)} \cdot \frac{\pi(Stick)}{\beta(Stick)}$
   $\rho = \frac{0.8}{0.5} \cdot \frac{0.8}{0.5} \cdot \frac{0.2}{0.5} = 1.6 \cdot 1.6 \cdot 0.4 = 2.56 \cdot 0.4 = 1.024$

4. **Sarsa vs. Q-learning Target:** Given a state $S'$ with action values $Q(S', a_1) = 10, Q(S', a_2) = 2, Q(S', a_3) = 6$. The current policy is $\epsilon$-greedy with $\epsilon=0.3$. Assume action $a_1$ is selected at $S'$ during an episode. Reward $R=5$ was received for the transition to $S'$. Calculate the TD target for the previous state-action pair using: a) Sarsa ($\gamma = 1.0$) b) Q-learning ($\gamma = 1.0$)

   a) **Sarsa:** Target = $R + \gamma Q(S', a_{selected})$
      Since action $a_1$ was selected: $Target = 5 + 1.0(10) = 15$
   b) **Q-learning:** Target = $R + \gamma \max_a Q(S', a)$
      $Target = 5 + 1.0(\max(10, 2, 6)) = 5 + 10 = 15$
      (Note: here they match because $a_1$ happened to be the greedy action).

5. **Weighted Importance Sampling:** Two episodes provide returns $G_1 = 10$ and $G_2 = 5$. Importance ratios for these episodes are $\rho_1 = 1.5$ and $\rho_2 = 0.5$. Calculate the weighted importance sampling estimate for the value of the state.

   $V = \frac{\rho_1 G_1 + \rho_2 G_2}{\rho_1 + \rho_2}$
   $V = \frac{1.5(10) + 0.5(5)}{1.5 + 0.5} = \frac{15 + 2.5}{2.0} = \frac{17.5}{2.0} = 8.75$
