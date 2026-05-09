# Numerical Questions - Lecture 4 (Monte Carlo & TD Learning)

1. **MC Return Calculation:**
   Given an episode: $S_0, A_0, R_1=2, S_1, A_1, R_2=-1, S_2, A_2, R_3=4, Terminal$.
   Assume $\gamma = 0.9$.
   Calculate the return $G_0$ observed from state $S_0$.

2. **TD(0) Update:**
   Current estimate $V(S_t) = 10.0$.
   The agent takes an action, receives reward $R_{t+1} = 2.0$ and transitions to $S_{t+1}$.
   The estimate for the next state is $V(S_{t+1}) = 5.0$.
   Using $\alpha = 0.1$ and $\gamma = 0.9$, calculate the updated value $V(S_t)$.

3. **Importance Sampling Ratio:**
   Target policy $\pi(a \mid s) = 0.8$ for action 'Hit', $0.2$ for 'Stick'.
   Behavior policy $b(a \mid s) = 0.5$ for action 'Hit', $0.5$ for 'Stick'.
   An episode follows the sequence: $Hit, Hit, Stick$.
   Calculate the importance sampling ratio $\rho_{0:2}$ for this episode.

4. **Sarsa vs. Q-learning Target:**
   Given a state $S'$ with action values $Q(S', a_1) = 10, Q(S', a_2) = 2, Q(S', a_3) = 6$.
   The current policy $\pi(S')$ is $\epsilon$-greedy with $\epsilon=0.3$.
   Assume action $a_1$ is selected at $S'$ during an episode.
   Reward $R=5$ was received for the transition to $S'$.
   Calculate the TD target for the previous state-action pair $(S, A)$ using:
   a) Sarsa ($\gamma = 1.0$)
   b) Q-learning ($\gamma = 1.0$)

5. **Weighted Importance Sampling:**
   Two episodes provide returns $G_1 = 10$ and $G_2 = 5$.
   Importance ratios for these episodes are $\rho_1 = 1.5$ and $\rho_2 = 0.5$.
   Calculate the weighted importance sampling estimate for the value of the state.
