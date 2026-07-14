---
layout: post
---

# Solutions to Subjective Questions (Lecture 9)

1. **Stochastic Policies:** If an agent plays Rock-Paper-Scissors deterministically (e.g., always playing Rock), an intelligent opponent will quickly learn to always play Paper and win 100% of the time. The optimal policy (Nash Equilibrium) is to play each option with exactly 33.3% probability. Q-learning struggles here because it seeks the single action with the maximum value. Policy Gradients directly output a probability distribution (via a Softmax layer), allowing them to naturally learn and represent true stochastic policies.
2. **The Variance Problem:** REINFORCE uses the full Monte Carlo return $G_t$ (the sum of all rewards until the end of the episode). This creates massive variance because $G_t$ depends on *every single random action and environment transition* that occurs after time step $t$. An excellent action at time $t$ might be penalized if the agent randomly makes a terrible mistake at time $t+10$. Actor-Critic methods solve this by bootstrapping: the Critic learns to estimate the expected return from the next state $V(S_{t+1})$. We replace the noisy, full-episode rollout $G_t$ with a 1-step TD Target $R_{t+1} + \gamma V(S_{t+1})$, drastically reducing variance at the cost of some initial bias.
