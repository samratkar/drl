---
tags : [drl-class-notes]
title : "004 - Sara's sojourn - Temporal Difference"
classes : ["2026-05-16"]
---

## The Step-by-Step Instinct (Temporal Difference)

Sara realizes that waiting until nightfall to learn is dangerous. She develops a new instinct: **Temporal Difference (TD)** learning.

She takes **one step**, receives an immediate reward $R_{t+1}$, and then uses her *previous guess* of the next state's value $V(S_{t+1})$ to update her current position. She is solving the **Bellman Optimality Equation** one step at a time:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma \max_a Q(S_{t+1}, a)}_{\text{TD Target}} - Q(S_t, A_t) \right]$$

**Key Insights:**
*   **The TD Target:** The "Target" is no longer the full map (DP) or the full day (MC). It is just the next time step.
*   **Bootstrapping + Sampling:** She is guessing based on a guess, but doing so while actually walking in the forest.