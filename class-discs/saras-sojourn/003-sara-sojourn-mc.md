---
tags : [drl-class-notes]
title : "003 - Sara's sojourn - Monte Carlo"
classes : ["2026-05-09"]
---

## The Lost Map (Monte Carlo)

The map is lost in a fire. Sara is now blind to the probabilities. She must learn by **Interacting**.

She decides to walk a full journey (an **Episode**) from her camp to the Great Waterfall and back. Only when the day ends does she look at her total berries ($G_t$) and update her memory. This is **Monte Carlo (MC)**:

$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

**Key Insights:**
*   **No Bootstrapping:** She doesn't use guesses. She waits for the "Ground Truth" $G_t$.
*   **High Variance:** If she got lucky one day, she might overvalue a path.
*   **Wait until the End:** She can't learn anything until the episode is over.