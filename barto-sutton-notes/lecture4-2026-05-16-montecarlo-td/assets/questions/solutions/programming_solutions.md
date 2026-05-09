---
layout: post
---

# Programming Question Solutions - Lecture 4

### 1. MC Policy Evaluation
The core logic involves generating a full episode and then iterating backward to compute the return $G$ for each state. Using a dictionary for $V$ allows handling large or sparse state spaces.

### 2. TD(0) Prediction
Unlike MC, TD(0) updates the value function $V$ after every transition. The target is the immediate reward plus the discounted value of the next state.

### 3. Sarsa (On-policy)
Sarsa is defined by the quintuple $(S, A, R, S', A')$. Crucially, the action $A'$ must be chosen using the *current* exploratory policy (e.g., $\epsilon$-greedy) and then *actually* executed in the next step.

### 4. Q-Learning (Off-policy)
Q-learning updates $Q(s, a)$ using the maximum $Q$-value of the next state, regardless of which action the agent will actually take. This allows it to learn the optimal policy while still exploring.

### 5. Importance Sampling Weight Update
Incremental update for Weighted IS:
```python
C = C + rho
V = V + (rho / C) * (G - V)
```
Where $C$ is the sum of importance ratios seen so far.

---
*Implementation code can be found in `solutions/programs/programming_implementations.py`.*
