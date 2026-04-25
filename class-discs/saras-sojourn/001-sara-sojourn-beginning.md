---
tags : [drl-class-notes]
title : "001 - Sara's sojourn - The beginning"
classes : ["2026-04-26"]
---

# Sara's sourjourn - Sara and the Infinite Forest: A Unified Foundation of Deep RL

This is a story of the adventures of Sara. She is a legendary explorer who has just discovered the "Infinite Forest" — a place where the **trees shift**, the rewards are sweet, and the logic of survival follows the laws of mathematics.

This document serves as both a narrative and a rigorous mathematical guide to the core principles of RL.


## The Arrival (The Markov Decision Process)

Sara stands at the entrance of the forest. She realizes that to survive, she needs a formal framework to describe her world. She defines her environment as a **Markov Decision Process (MDP)**, a mathematical tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

*   **States ($\mathcal{S}$):** Her current location (e.g., "The Mossy Rock").
*   **Actions ($\mathcal{A}$):** Her choices (e.g., "Walk North", "Climb Tree").
*   **Transition Dynamics ($\mathcal{P}$):** The probability $p(s', r | s, a)$ that taking action $a$ in state $s$ leads to state $s'$ and reward $r$. This is the "Model" or the "Rules of the Game."
*   **Rewards ($\mathcal{R}$):** The immediate feedback (e.g., berries).
*   **Discount Factor ($\gamma$):** How much she values future rewards ($0 \le \gamma \le 1$).
*   **Policy ($\pi$):** Her strategy, a mapping from states to actions, $\pi(a | s)$.
*   **Return ($G_t$):** The total discounted reward she expects to receive starting from time $t$.
*   **Value Functions ($V$ and $Q$):** Her "feelings" about states and actions, which we will define later.
*   **Optimality:** The quest to find the best policy $\pi^*$ that maximizes her return.
*   **Bellman Equations:** The recursive relationships that define the value functions under a given policy or the optimal policy.
  

## The Winter Hunger (Return $G$ and Discount $\gamma$)

Sara finds a handful of berries ($R=1$). She is happy, but then she thinks about the coming winter. She doesn't just want berries *now*; she wants the maximum possible berries over her entire life. This total sum is her **Return ($G_t$)**:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**The Intuition:** The return is the "Ground Truth" goal. Without $\gamma$, the sum might go to infinity in a never-ending forest, making it impossible to compare two paths. $\gamma$ ensures Sara is "impatient" enough to survive.


## The Clearing vs. The Path (V vs. Q)

Sara reaches a sunlit clearing. She realizes she needs to value her life in two different ways:

1.  **State-Value ($V$):** "How good is it to be in this clearing?" 
    $$V_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$
2.  **Action-Value ($Q$):** "How good is the *action* of entering the cave from here?"
    $$Q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

**The Interrelationship:** Sara's "feeling" about the clearing ($V$) is simply the weighted average of the value of all the paths ($Q$) she might take, based on her current strategy (**Policy $\pi$**):
$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$




