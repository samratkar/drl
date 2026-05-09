# Numerical Questions - Lecture 5 (n-step Bootstrapping & Planning)

1. **Calculating n-step Returns:**
   Given a sequence of rewards $R_1=2, R_2=-1, R_3=4, R_4=10$ and a sequence of states $S_0, S_1, S_2, S_3, S_4$. 
   Assume $\gamma = 0.9$ and the current estimate for the value of state $S_4$ is $V(S_4) = 5.0$.
   Calculate:
   a) The 1-step return $G_{0:1}$
   b) The 2-step return $G_{0:2}$
   c) The 3-step return $G_{0:3}$
   d) The 4-step return $G_{0:4}$

2. **Importance Sampling Ratio:**
   Consider an off-policy n-step task.
   Target policy $\pi(a \mid s)$: $\pi(left \mid s) = 0.8, \pi(right \mid s) = 0.2$.
   Behavior policy $b(a \mid s)$: $b(left \mid s) = 0.5, b(right \mid s) = 0.5$.
   In an episode, the agent takes the following actions: $A_0=left, A_1=left, A_2=right$.
   Calculate the 3-step importance sampling ratio $\rho_{0:2}$.

3. **Dyna-Q+ Exploration Bonus:**
   In Dyna-Q+, the bonus added to the reward in planning is $r + \kappa\sqrt{\tau}$.
   If $\kappa = 0.01$ and a state-action pair $(s, a)$ was last tried 1000 time steps ago, what is the exploration bonus added to the simulated reward?

4. **Tree Backup Returns:**
   In a 2-step Tree Backup update, we visit state $S_t$, take action $A_t$, reach $S_{t+1}$, then take $A_{t+1}$ and reach $S_{t+2}$.
   Assume $\gamma=1.0$, $R_{t+1}=2, R_{t+2}=3$.
   Policy $\pi(a \mid S_{t+1})$: $A_{t+1}$ has prob 0.6, other action $a'$ has prob 0.4.
   Estimates: $Q(S_{t+1}, a') = 10, Q(S_{t+2}, a) = 5$ for all $a$ in $S_{t+2}$.
   Calculate the 2-step Tree Backup return $G_{t:t+2}$.

5. **Wait-and-See return:**
   For a task with $n=3, \gamma=0.9$.
   Rewards: $R_1=1, R_2=1, R_3=1$.
   The episode ends at $T=3$.
   What is the n-step return $G_{0:3}$? (Hint: Does it involve bootstrapping?)
