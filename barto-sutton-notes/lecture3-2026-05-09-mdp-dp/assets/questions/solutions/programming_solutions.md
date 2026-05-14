---
layout: post
---

# Solutions - Programming Questions

1. **Dynamics Representation**: How would you represent the dynamics $p(s', r \mid s, a)$ in a Python dictionary for a small MDP?

   **Answer:**
```python
# p[s][a] = [(prob, next_state, reward), ...]
P = {
    'cool': {
        'slow': [(1.0, 'cool', 1)],
        'fast': [(0.5, 'cool', 2), (0.5, 'warm', 2)]
    },
    # ...
}
```

2. **Policy Definition**: Write a Python function `get_action(policy, state)` that handles both deterministic policies (dictionary mapping `state -> action`) and stochastic policies (dictionary mapping `state -> {action: probability}`).

   **Answer:**
```python
def get_action(policy, state):
    action_data = policy[state]
    if isinstance(action_data, dict): # Stochastic
        actions = list(action_data.keys())
        probs = list(action_data.values())
        return np.random.choice(actions, p=probs)
    return action_data # Deterministic
```

3. **In-place Update**: What is the Python code for an "in-place" update of a value function $V[s]$ during Policy Evaluation?

   **Answer:**
```python
V[s] = sum(prob * (reward + gamma * V[next_state]) 
           for prob, next_state, reward in P[s][policy[s]])
```

4. **Termination Condition**: Write a `while` loop condition that continues Policy Evaluation until the maximum change in value across all states is less than a threshold `theta`.

   **Answer:**
```python
while True:
    delta = 0
    for s in states:
        v_old = V[s]
        # ... update V[s] ...
        delta = max(delta, abs(v_old - V[s]))
    if delta < theta:
        break
```

5. **Policy Improvement Logic**: Given a value function $V$ and a model $P$, write the logic to find the greedy action for state $s$.

   **Answer:**
```python
def get_greedy_action(s, V, P, gamma):
    action_values = []
    for a in actions:
        q = sum(prob * (reward + gamma * V[next_state]) 
                for prob, next_state, reward in P[s][a])
        action_values.append(q)
    return actions[np.argmax(action_values)]
```

6. **Value Iteration Core**: Write the single line of code inside the state sweep of Value Iteration that updates $V[s]$.

   **Answer:**
```python
V[s] = max(sum(p * (r + gamma * V[s_next]) for p, s_next, r in P[s][a]) for a in actions)
```

7. **Jack's Car Rental (Poisson)**: Why is it computationally efficient to pre-compute Poisson probabilities $P(n, \lambda)$ before starting DP? Write a snippet to pre-compute these up to a maximum number of cars.

   **Answer:** Pre-computing is faster because the same Poisson values are used in every state/action update across many iterations.
```python
import math
def poisson(n, lam):
    return (lam**n / math.factorial(n)) * math.exp(-lam)

poisson_cache = {lam: [poisson(n, lam) for n in range(21)] for lam in [3, 4, 2]}
```

8. **Gambler's Problem Stakes**: Write a function `get_possible_stakes(s, goal)` that returns a list of valid actions for capital `s`.

   **Answer:**
```python
def get_possible_stakes(s, goal):
    return list(range(1, min(s, goal - s) + 1))
```

9. **Tie-breaking**: In policy improvement, if multiple actions have the same maximum Q-value, how would you modify your `argmax` to ensure the policy remains stable or explores fairly?

   **Answer:** Use `np.where` to find all indices with the maximum value and pick one randomly or pick the first one consistently to ensure policy stability.
```python
indices = np.where(action_values == np.max(action_values))[0]
return actions[indices[0]] # Consistent tie-breaking
```

10. **State Mapping**: If you have a 4x4 grid, write a lambda function to convert a 2D coordinate `(r, c)` to a 1D index `i`.

    **Answer:**
```python
grid_to_id = lambda r, c, width: r * width + c
```

11. **Absorbing State Implementation**: How do you represent a terminal state in a transition matrix or dynamics dictionary to ensure $V(\text{terminal}) = 0$ is maintained?

    **Answer:**
```python
# Transition from terminal state always goes to itself with 0 reward
P['terminal'] = {'any_action': [(1.0, 'terminal', 0)]}
# Or simply exclude it from the update loop and initialize V['terminal'] = 0.
```

12. **Asynchronous DP (Random)**: Write a code snippet that selects a random state to update, rather than doing a systematic sweep.

    **Answer:**
```python
import random
s = random.choice(states)
update_value(s)
```

13. **Model-Based vs. Model-Free**: In code, how does the input to a DP algorithm (like Value Iteration) differ from the input to a model-free algorithm (like Q-Learning)?

    **Answer:** DP requires the full transition model $P(s', r \mid s, a)$ as input. Model-free algorithms require an environment `step(action)` function that returns $(s', r, \text{done})$.

14. **Efficiency in Summation**: When computing $\sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$, how can you optimize the code if most transitions have a probability of zero?

    **Answer:** Use a list of non-zero transitions for each $(s, a)$ instead of iterating over the entire state space $S$.

15. **Discount Factor Handling**: In a function `calculate_q_value(s, a, V, P, gamma)`, where should `gamma` be applied?

    **Answer:** It should multiply the value of the *next* state: `reward + gamma * V[next_state]`.

16. **Verifying Convergence**: Write a function `is_optimal(V, P, gamma, theta)` that returns `True` if the Bellman Optimality Equation is satisfied for all states within `theta`.

    **Answer:**
```python
def is_optimal(V, P, gamma, theta):
    for s in states:
        best_q = max(sum(p*(r + gamma*V[sn]) for p, sn, r in P[s][a]) for a in actions)
        if abs(V[s] - best_q) > theta:
            return False
    return True
```

17. **Handling Large Action Spaces**: If the action space is continuous, how would you modify the `max_a` step in Value Iteration?

    **Answer:** Discretize the action space or use an optimization library (like `scipy.optimize`) to find the maximum.

18. **Recycling Robot States**: Define an Enum class for the Recycling Robot states to make the code more readable.

    **Answer:**
```python
from enum import Enum
class RobotState(Enum):
    HIGH = 0
    LOW = 1
```

19. **Policy Stability Check**: Write a function `are_policies_equal(pi1, pi2)` that compares two policies represented as dictionaries.

    **Answer:**
```python
def are_policies_equal(pi1, pi2):
    return all(pi1[s] == pi2[s] for s in states)
```

20. **Visualizing the Value Function**: Given a 1D array `V` representing a 4x4 grid, write the code to reshape and print it as a formatted matrix.

    **Answer:**
```python
print(V.reshape(4, 4))
```
