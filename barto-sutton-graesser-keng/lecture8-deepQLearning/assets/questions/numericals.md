---
layout: post
---

# Numerical Questions (Lecture 8: Deep Q-Learning)

**Question 1: Target Calculation (DQN vs DDQN)**
At time step $t$, the agent transitions from $s$ to $s'$ and receives a reward $R = 5$. The discount factor is $\gamma = 0.9$. 
There are 3 possible actions: $a_1, a_2, a_3$.
The Online Network $\theta$ outputs the following Q-values for $s'$:
* $Q(s', a_1; 	heta) = 10$
* $Q(s', a_2; 	heta) = 20$
* $Q(s', a_3; 	heta) = 15$

The Target Network $\theta'$ outputs the following Q-values for $s'$:
* $Q(s', a_1; 	heta') = 18$
* $Q(s', a_2; 	heta') = 14$
* $Q(s', a_3; 	heta') = 25$

1. Calculate the TD Target $y_j$ if the agent is using standard **DQN**.
2. Calculate the TD Target $y_j$ if the agent is using **Double DQN (DDQN)**.
3. Explain the difference in the result.
