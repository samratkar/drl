---
layout: post
---

# Programming Questions - Lecture 4 (Monte Carlo & TD Learning)

1. **MC Policy Evaluation:**
   Implement a Python function `first_visit_mc_prediction(env, policy, episodes, gamma)` that evaluates a given policy in a discrete environment.
   - The environment should provide an `env.reset()` and `env.step(action)` method.
   - Return the estimated value function $V$ as a dictionary or array.

2. **TD(0) Prediction:**
   Implement a Python function `td_0_prediction(env, policy, alpha, episodes, gamma)` that uses temporal-difference learning to estimate $V_\pi$.
   - Update $V$ after every step.
   - Ensure the function handles terminal states correctly.

3. **Sarsa (On-policy Control):**
   Implement the Sarsa algorithm: `sarsa_control(env, alpha, epsilon, episodes, gamma)`.
   - Maintain an action-value table $Q(s, a)$.
   - Use $\epsilon$-greedy action selection for both the current and the next step.

4. **Q-Learning (Off-policy Control):**
   Implement the Q-learning algorithm: `q_learning_control(env, alpha, epsilon, episodes, gamma)`.
   - Update $Q$ using the maximum value of the next state.
   - Ensure the policy used for interaction is $\epsilon$-greedy.

5. **Importance Sampling Weight Update:**
   Implement a small snippet to update the weighted importance sampling estimate incrementally.
   - Given the current estimate $V$, cumulative weight sum $C$, importance ratio $\rho$, and return $G$:
   - Calculate the new $V$ and new $C$.
