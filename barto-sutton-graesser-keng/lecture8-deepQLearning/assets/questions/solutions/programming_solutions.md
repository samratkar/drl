# Solutions to Programming Questions (Lecture 8)

**Question 1 Solution:**
```python
import torch

def compute_ddqn_target(rewards, dones, gamma, online_q_next, target_q_next):
    # 1. Action Selection: Get the index of the max Q-value from the Online Network
    best_actions = online_q_next.argmax(dim=1, keepdim=True)
    
    # 2. Action Evaluation: Gather the Q-values from the Target Network using those selected actions
    evaluated_q_values = target_q_next.gather(1, best_actions)
    
    # 3. Compute Target: R + gamma * Q_eval (zeroing out future rewards if state is terminal)
    y_j = rewards + gamma * evaluated_q_values * (~dones)
    
    return y_j
```
