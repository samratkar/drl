---
layout: post
---

# Solutions to Programming Questions (Lecture 5)

**Question 1 Solution:**

```python
# 1. Q-learning Update (Off-Policy)
def update_q_learning(Q, state, action, reward, next_state, alpha, gamma, env):
    # Find the maximum Q-value in the next state
    max_next_q = max([Q[next_state][a] for a in env.action_space])
    
    # Calculate TD Target
    td_target = reward + gamma * max_next_q
    
    # Calculate TD Error
    td_error = td_target - Q[state][action]
    
    # Update Q-table
    Q[state][action] += alpha * td_error


# 2. SARSA Update (On-Policy)
def update_sarsa(Q, state, action, reward, next_state, next_action, alpha, gamma):
    # Use the Q-value of the actual action selected for the next state
    next_q = Q[next_state][next_action]
    
    # Calculate TD Target
    td_target = reward + gamma * next_q
    
    # Calculate TD Error
    td_error = td_target - Q[state][action]
    
    # Update Q-table
    Q[state][action] += alpha * td_error
```
