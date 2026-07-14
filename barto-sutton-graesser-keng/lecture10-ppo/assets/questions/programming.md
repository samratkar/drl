---
layout: post
---

# Programming Questions (Lecture 10: PPO)

**Question 1: Implementing the PPO Clipped Loss in PyTorch**
You are given the following PyTorch tensors (all of shape `(batch_size,)`):
* `ratio`: The probability ratio $r_t(\theta)$
* `adv`: The Advantage $A_t$
* `epsilon`: A scalar float (e.g., 0.2)

Write a PyTorch function to compute the exact PPO Clipped Surrogate Objective. 
*(Hint: Use `torch.clamp()` and `torch.min()`)*
