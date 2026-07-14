---
layout: post
---

# Solutions to Programming Questions (Lecture 9)

**Question 1 Solution:**
```python
import torch

def compute_reinforce_loss(action_probs, returns):
    # 1. Calculate the log probabilities: ln(pi)
    log_probs = torch.log(action_probs)
    
    # 2. Multiply by the observed returns
    objective = log_probs * returns
    
    # 3. Sum over the trajectory to get the total objective J(theta)
    J_theta = torch.sum(objective)
    
    # 4. PyTorch minimizes loss via Gradient Descent. 
    # To maximize J_theta, we must minimize negative J_theta!
    loss = -J_theta
    
    return loss
```
