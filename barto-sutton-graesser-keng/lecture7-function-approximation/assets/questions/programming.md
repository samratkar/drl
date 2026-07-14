---
layout: post
---

# Programming Questions (Chapter 9: Function Approximation)

**Question 1: Implementing Semi-gradient TD(0)**
Write a Python function `semi_gradient_td_update(w, x_t, r, x_next, alpha, gamma)` that performs a single weight update for linear semi-gradient TD(0). 
Assume `w`, `x_t`, and `x_next` are numpy arrays. The function should return the updated weight vector `w_new`.
