---
layout: post
---

# Solutions to MCQs (Lecture 10)

1. **b) A single excessively large gradient step can destroy the policy, causing fatal unrecoverable divergence.** (Standard PG steps too far, breaking the policy and causing the agent to gather garbage data).
2. **b) To prevent the new policy from diverging too far from the old policy in a single update.** (This is the definition of the Trust Region constraint).
3. **b) To encourage exploration by penalizing the network if it becomes too certain (deterministic) too early.** (High entropy means high randomness. Rewarding it prevents premature convergence).
4. **c) The clipping function activates ($r_t < 0.8$), and the gradient becomes $0$, stopping the update.** (Because $0.5$ is below the $1-\epsilon$ threshold of $0.8$, the $\min()$ operator selects the clipped bound, killing the gradient).
