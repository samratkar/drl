---
layout: post
---

# Programming Questions (Lecture 5: Temporal Difference Learning)

**Question 1: Implementing the SARSA and Q-Learning Updates**
You are writing a standard tabular RL agent in Python. You have a Q-table stored as a dictionary: `Q[state][action]`. 
Assume you just experienced the transition: `(state, action, reward, next_state)`.

1. Write the Python code to perform a single **Q-learning** update. Assume you have `alpha` (learning rate), `gamma` (discount factor), and access to all actions via `env.action_space`.
2. Write the Python code to perform a single **SARSA** update. Assume the agent has already chosen `next_action` using its $\epsilon$-greedy policy.
