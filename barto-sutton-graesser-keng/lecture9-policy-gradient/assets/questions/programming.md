---
layout: post
---

# Programming Questions (Lecture 9: Policy Gradients)

**Question 1: Implementing the REINFORCE Loss in PyTorch**
You are given the following PyTorch tensors representing an entire trajectory rollout:
* `action_probs`: A tensor of shape `(seq_len,)` containing the specific probabilities $\pi(A_t|S_t, \theta)$ of the actions the agent *actually took* at each step.
* `returns`: A tensor of shape `(seq_len,)` containing the calculated total returns $G_t$ for each step.

Write the PyTorch code to compute the REINFORCE loss function.
*(Hint: Remember that PyTorch optimizers do Gradient Descent, but we want to do Gradient Ascent to maximize the objective).*
