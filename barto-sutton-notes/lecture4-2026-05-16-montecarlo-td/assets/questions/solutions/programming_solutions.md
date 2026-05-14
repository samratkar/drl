---
layout: post
---

# Programming Question Solutions - Lecture 4

### 1. MC Policy Evaluation

**Question:** Implement a Python function `first_visit_mc_prediction(env, policy, episodes, gamma)` that evaluates a given policy in a discrete environment. The environment should provide an `env.reset()` and `env.step(action)` method. Return the estimated value function $V$ as a dictionary or array.

The core logic involves generating a full episode and then iterating backward to compute the return $G$ for each state. Using a dictionary for $V$ allows handling large or sparse state spaces.

### 2. TD(0) Prediction

**Question:** Implement a Python function `td_0_prediction(env, policy, alpha, episodes, gamma)` that uses temporal-difference learning to estimate $V_\pi$. Update $V$ after every step. Ensure the function handles terminal states correctly.

Unlike MC, TD(0) updates the value function $V$ after every transition. The target is the immediate reward plus the discounted value of the next state.

### 3. Sarsa (On-policy Control)

**Question:** Implement the Sarsa algorithm: `sarsa_control(env, alpha, epsilon, episodes, gamma)`. Maintain an action-value table $Q(s, a)$. Use $\epsilon$-greedy action selection for both the current and the next step.

Sarsa is defined by the quintuple $(S, A, R, S', A')$. Crucially, the action $A'$ must be chosen using the *current* exploratory policy (e.g., $\epsilon$-greedy) and then *actually* executed in the next step.

### 4. Q-Learning (Off-policy Control)

**Question:** Implement the Q-learning algorithm: `q_learning_control(env, alpha, epsilon, episodes, gamma)`. Update $Q$ using the maximum value of the next state. Ensure the policy used for interaction is $\epsilon$-greedy.

Q-learning updates $Q(s, a)$ using the maximum $Q$-value of the next state, regardless of which action the agent will actually take. This allows it to learn the optimal policy while still exploring.

### 5. Importance Sampling Weight Update

**Question:** Implement a small snippet to update the weighted importance sampling estimate incrementally. Given the current estimate $V$, cumulative weight sum $C$, importance ratio $\rho$, and return $G$: Calculate the new $V$ and new $C$.

Incremental update for Weighted IS:
```python
C = C + rho
V = V + (rho / C) * (G - V)
```
Where $C$ is the sum of importance ratios seen so far.

---
*Implementation code can be found in `solutions/programs/programming_implementations.py`.*
