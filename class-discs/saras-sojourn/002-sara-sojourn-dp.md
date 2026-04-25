---
tags : [drl-class-notes]
title : "002 - Sara's sojourn - Dynamic Programming"
classes : ["2026-05-02"]
---

## The Ranger’s Map (Dynamic Programming)

Sara finds a "perfect map" (the Model $\mathcal{P}$). It tells her every probability $p(s', r | s, a)$. 

Crucially, **RL is not only about interaction.** Because Sara has the map, she can perform **Dynamic Programming (DP)**—an offline planning process. She sits by her campfire and solves the **Bellman Expectation Equation** recursively:

$$V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_\pi(s') \right]$$

**Key Insights:**
*   **Planning, not Interacting:** Sara isn't walking; she is calculating. 
*   **Full Backup:** She looks at *every* possible future state the map says is possible.
*   **Bootstrapping:** She updates her guess of $V(s)$ using her guess of $V(s')$.
*   **No $\alpha$:** Because the map is "truth," she sets $\alpha=1$ (effectively), updating the value immediately to match the map's prediction.