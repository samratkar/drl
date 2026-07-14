---
layout: post
---

# Subjective Questions (Lecture 10: PPO)

1. **Sample Efficiency:** Explain why standard REINFORCE is considered "sample inefficient", and exactly how PPO's clipped objective allows it to be much more sample efficient.
2. **The Minimum Operator:** The PPO surrogate objective uses a $\min()$ operator between the unclipped and clipped ratios: $\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$. Explain why we use a *minimum* bound rather than just exclusively using the clipped value. 
