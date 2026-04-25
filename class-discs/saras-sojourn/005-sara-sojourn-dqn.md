---
tags : [drl-class-notes]
title : "005 - Sara's sojourn - Deep Q-Learning"
classes : ["2026-05-23"]
---

## The Great Fog (Deep Q-Learning / DQN)

The forest expands to trillions of trees. Sara cannot remember a Q-value for every single tree. 

A thick fog rolls in. She can no longer see coordinates; she only sees **Features** (e.g., "tall pine", "smell of pine"). She needs a **Neural Network (Brain)** to approximate the Q-function: $Q(s, a; \theta)$.

To train her brain, she minimizes the **Loss Function**, which is the squared difference between her "Step-by-Step Instinct" (TD Target) and her "Neural Prediction":

$$Loss(\theta) = \mathbb{E} \left[ \left( \underbrace{R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)}_{\text{Target (Truth from experience)}} - \underbrace{Q(S_t, A_t; \theta)}_{\text{Prediction (Current Brain State)}} \right)^2 \right]$$




## Summary: The Evolutionary Thread

| Mode | Sara's Action | Math Target ($U_t$) | Bootstraps? | Updates |
| :--- | :--- | :--- | :--- | :--- |
| **DP** | Planning by campfire | $\mathbb{E}[R + \gamma V(s')]$ | **Yes** | Anytime (Offline) |
| **MC** | Reviewing the full day | $G_t$ (Full Return) | **No** | After Episode |
| **TD** | Learning as she walks | $R + \gamma V(s')$ | **Yes** | After Every Step |
| **DQN** | Pattern matching in fog | TD Target + NN | **Yes** | After Every Step |

Sara realizes that while her methods changed—from a map, to a day-review, to a step-instinct, to a complex brain—she was always chasing the same ghost: **The Bellman Optimality Equation.**
