---
layout: post
---

# Solutions to Programming Questions (Chapter 9)

**Question 1 Solution:**
```python
import numpy as np

def semi_gradient_td_update(w, x_t, r, x_next, alpha, gamma):
    """
    Performs a single Semi-gradient TD(0) update for a linear approximator.
    
    Args:
        w (np.array): Current weight vector.
        x_t (np.array): Feature vector for current state S_t.
        r (float): Reward received after transition.
        x_next (np.array): Feature vector for next state S_{t+1}.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        
    Returns:
        np.array: The updated weight vector.
    """
    # 1. Calculate current value estimate
    v_t = np.dot(w, x_t)
    
    # 2. Calculate next value estimate
    v_next = np.dot(w, x_next)
    
    # 3. Calculate TD target and TD error
    td_target = r + gamma * v_next
    td_error = td_target - v_t
    
    # 4. Update weights (Gradient is exactly x_t for linear approximation)
    w_new = w + alpha * td_error * x_t
    
    return w_new
```
