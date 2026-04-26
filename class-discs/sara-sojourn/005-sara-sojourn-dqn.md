---
tags : [drl-class-notes]
title : "005 - Sara's sojourn - Deep Q-Learning"
classes : ["2026-05-23"]
---

## The Great Fog and the Brain (Deep Q-Networks)

Sara had mastered the frozen lake using Temporal Difference. She could step, learn, and survive. But one day, the lake expanded into an endless ocean of ice. Trillions of unique ice patches stretched before her. 

She could no longer memorize a table (a $Q$-table) for every single coordinate. Furthermore, a thick fog rolled in. She could no longer see her exact coordinates; she could only sense **Features** (e.g., the speed of the wind, the angle of the ice, the smell of the waterfall).

Sara needed something better than a table. She needed a **Brain**.

### 1. Function Approximation: The Neural Network

Instead of a table where $Q(s, a)$ is a specific number written on a piece of paper, Sara builds a **Neural Network**. 

She feeds her sensory features (the state $s$) into the network, and the network outputs its best guess for the $Q$-values of all possible actions: $Q(s, a; \theta)$, where $\theta$ represents the connections (weights) inside her brain.

**The shift:**
*   **Old Way (Tabular):** Look up row 4, column 2. The answer is 0.5.
*   **New Way (Deep RL):** Calculate `0.2*wind + 0.8*angle - 0.1*smell`. The answer is roughly 0.5.

Because it calculates based on features, if Sara sees a *new* patch of ice that looks and feels like an *old* patch of ice, her brain can **Generalize**. She doesn't have to learn from scratch.

### 2. The Loss Function: Chasing the Ghost

How does she train this brain? She uses the same instinct she developed in TD Learning: she compares her current guess to her "Step-by-Step" target.

She wants to minimize the **Loss** (the squared difference) between her "TD Target" (the truth from experience) and her "Prediction" (what her brain originally guessed):

$$Loss(\theta) = \left( \underbrace{R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta)}_{\text{Target (Experience)}} - \underbrace{Q(S_t, A_t; \theta)}_{\text{Prediction (Current Brain)}} \right)^2$$

### 3. The Two Curses of Deep RL

When Sara first tried this, she went crazy. The network forgot things, fluctuated wildly, and crashed. She discovered two major problems when mixing Neural Networks with RL:

#### Curse 1: Correlated Data
If Sara walks in a straight line, she sees: Ice $\to$ Ice $\to$ Ice. If she trains her brain on these consecutive steps, her brain over-adapts to "Ice" and completely forgets what "Snow" looks like. The data is too correlated.

**The Cure: Experience Replay**
Sara starts carrying a **Journal**. Every time she takes a step $(S, A, R, S')$, she writes it down in her journal. 
At night, instead of learning from her most recent steps, she randomly opens her journal, picks a handful of random past memories (a "mini-batch"), and trains her brain on those. This breaks the correlation and reminds her brain of all the different things she has seen.

#### Curse 2: The Moving Target
Look at the Loss Function. The "Target" is $R + \gamma \max Q(S', a'; \theta)$. 
But wait... $\theta$ is inside the target! Every time Sara updates her brain ($\theta$) to get closer to the target, the target itself moves! It's like a dog chasing its own tail.

**The Cure: The Target Network**
Sara creates a *second* brain: the **Target Network** ($\theta^-$). 
She uses her main brain ($\theta$) to walk and explore. But when she calculates the "TD Target", she uses the frozen, older Target Network ($\theta^-$). 
Every few days, she copies her main brain over to the Target Network to update it. This keeps the target stationary long enough for the main brain to catch up.

$$Loss(\theta) = \left( R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-) - Q(S_t, A_t; \theta) \right)^2$$

### 4. The Deep Q-Network (DQN) Algorithm

Here is Sara's final, complex survival system for infinite, foggy worlds:

1.  Initialize **Main Brain** $Q(s, a; \theta)$.
2.  Initialize **Target Brain** $Q(s, a; \theta^-) = Q(s, a; \theta)$.
3.  Initialize an empty **Replay Journal**.
4.  For every step:
    *   Choose $A$ using $\epsilon$-greedy from the Main Brain.
    *   Take step, observe $R, S'$.
    *   Save $(S, A, R, S')$ to the Journal.
    *   **Sample a random mini-batch** from the Journal.
    *   Calculate the TD Target using the **Target Brain** ($\theta^-$).
    *   Perform a Gradient Descent step on the **Main Brain** ($\theta$) to minimize the Loss.
    *   Every $C$ steps, sync the Target Brain to the Main Brain.

### Summary: The Evolutionary Thread

Sara realizes that while her methods evolved drastically, she was always chasing the same mathematical ghost: **The Bellman Optimality Equation.**

| Mode | Sara's Action | Math Target ($U_t$) | Bootstraps? | State Space |
| :--- | :--- | :--- | :--- | :--- |
| **DP** | Planning by campfire | $\mathbb{E}[R + \gamma V(s')]$ | **Yes** | Tiny (Tabular) |
| **MC** | Reviewing the full day | $G_t$ (Full Return) | **No** | Small (Tabular) |
| **TD** | Learning as she walks | $R + \gamma V(s')$ | **Yes** | Small (Tabular) |
| **DQN** | Pattern matching in fog | $R + \gamma \max Q(s'; \theta^-)$ | **Yes** | Infinite (Continuous) |

*Next: How does Sara handle continuous actions, like the precise tension of a bowstring? (Policy Gradients)*
