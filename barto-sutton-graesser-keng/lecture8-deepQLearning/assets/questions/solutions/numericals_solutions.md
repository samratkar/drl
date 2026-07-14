---
layout: post
---

# Solutions to Numerical Questions (Lecture 8)

**Question 1 Solution:**
1. **Standard DQN Target:**
   Standard DQN uses the Target Network $\theta'$ for both selection and evaluation:
   $y_j^{\text{DQN}} = R + \gamma \max_{a} Q(s', a; 	heta')$
   The maximum value from the Target Network is for $a_3$, which is $25$.
   $y_j^{\text{DQN}} = 5 + 0.9(25) = 5 + 22.5 = 27.5$

2. **Double DQN (DDQN) Target:**
   DDQN uses the Online Network $\theta$ for selection, and the Target Network $\theta'$ for evaluation.
   *Selection:* $	ext{argmax}_{a} Q(s', a; 	heta)$ -> The maximum value from the Online Network is for $a_2$ (value $20$).
   *Evaluation:* What is the value of $a_2$ according to the Target Network? $Q(s', a_2; 	heta') = 14$.
   $y_j^{\text{DDQN}} = R + \gamma Q(s', 	ext{argmax}_{a} Q(s', a; 	heta); 	heta')$
   $y_j^{\text{DDQN}} = 5 + 0.9(14) = 5 + 12.6 = 17.6$

3. **Explanation:**
   Notice that DQN yielded a much higher target ($27.5$) than DDQN ($17.6$). Standard DQN looked at the Target Network and blindly grabbed the highest value ($25$). However, the Online Network believed that action was quite poor ($15$). DDQN forced the algorithm to evaluate the action the Online policy actually preferred ($a_2$), resulting in a much more conservative and realistic target.
