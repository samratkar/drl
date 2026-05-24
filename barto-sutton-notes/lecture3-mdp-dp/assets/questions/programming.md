---
layout: post
---

# Programming Questions - MDPs & Dynamic Programming

1. **Dynamics Representation**: How would you represent the dynamics $p(s', r \mid s, a)$ in a Python dictionary for a small MDP?

2. **Policy Definition**: Write a Python function `get_action(policy, state)` that handles both deterministic policies (dictionary mapping `state -> action`) and stochastic policies (dictionary mapping `state -> {action: probability}`).

3. **In-place Update**: What is the Python code for an "in-place" update of a value function $V[s]$ during Policy Evaluation?

4. **Termination Condition**: Write a `while` loop condition that continues Policy Evaluation until the maximum change in value across all states is less than a threshold `theta`.

5. **Policy Improvement Logic**: Given a value function $V$ and a model $P$, write the logic to find the greedy action for state $s$.

6. **Value Iteration Core**: Write the single line of code inside the state sweep of Value Iteration that updates $V[s]$.

7. **Jack's Car Rental (Poisson)**: Why is it computationally efficient to pre-compute Poisson probabilities $P(n, \lambda)$ before starting DP? Write a snippet to pre-compute these up to a maximum number of cars.

8. **Gambler's Problem Stakes**: Write a function `get_possible_stakes(s, goal)` that returns a list of valid actions for capital `s`.

9. **Tie-breaking**: In policy improvement, if multiple actions have the same maximum Q-value, how would you modify your `argmax` to ensure the policy remains stable or explores fairly?

10. **State Mapping**: If you have a 4x4 grid, write a lambda function to convert a 2D coordinate `(r, c)` to a 1D index `i`.

11. **Absorbing State Implementation**: How do you represent a terminal state in a transition matrix or dynamics dictionary to ensure $V(\text{terminal}) = 0$ is maintained?

12. **Asynchronous DP (Random)**: Write a code snippet that selects a random state to update, rather than doing a systematic sweep.

13. **Model-Based vs. Model-Free**: In code, how does the input to a DP algorithm (like Value Iteration) differ from the input to a model-free algorithm (like Q-Learning)?

14. **Efficiency in Summation**: When computing $\sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$, how can you optimize the code if most transitions have a probability of zero?

15. **Discount Factor Handling**: In a function `calculate_q_value(s, a, V, P, gamma)`, where should `gamma` be applied?

16. **Verifying Convergence**: Write a function `is_optimal(V, P, gamma, theta)` that returns `True` if the Bellman Optimality Equation is satisfied for all states within `theta`.

17. **Handling Large Action Spaces**: If the action space is continuous, how would you modify the `max_a` step in Value Iteration?

18. **Recycling Robot States**: Define an Enum class for the Recycling Robot states to make the code more readable.

19. **Policy Stability Check**: Write a function `are_policies_equal(pi1, pi2)` that compares two policies represented as dictionaries.

20. **Visualizing the Value Function**: Given a 1D array `V` representing a 4x4 grid, write the code to reshape and print it as a formatted matrix.
