# Solutions to Programming Questions (Lecture 10)

**Question 1 Solution:**
```python
import torch

def compute_ppo_loss(ratio, adv, epsilon=0.2):
    # 1. Calculate unclipped objective
    unclipped_obj = ratio * adv
    
    # 2. Calculate clipped objective
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped_obj = clipped_ratio * adv
    
    # 3. Take the pessimistic bound (minimum)
    # PyTorch torch.min() computes element-wise minimum of two tensors
    surrogate_obj = torch.min(unclipped_obj, clipped_obj)
    
    # 4. PPO maximizes the surrogate, so to use PyTorch gradient descent, 
    # we return the negative mean of the batch
    loss = -surrogate_obj.mean()
    
    return loss
```
