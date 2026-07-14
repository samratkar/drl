---
layout: post
---

# Multiple Choice Questions (Lecture 10: PPO)

1. What is the primary problem with standard Policy Gradient methods that PPO attempts to solve?
   a) They cannot handle continuous action spaces.
   b) A single excessively large gradient step can destroy the policy, causing fatal unrecoverable divergence.
   c) The critic network overestimates Q-values.
   d) The learning rate decays too quickly.

2. In the PPO algorithm, why is the probability ratio $r_t(\theta)$ constrained within $[1-\epsilon, 1+\epsilon]$?
   a) To ensure the policy remains deterministic.
   b) To prevent the new policy from diverging too far from the old policy in a single update.
   c) To force the Advantage to always be positive.
   d) To ensure probabilities always sum exactly to 1.

3. In a shared-backbone PPO architecture, why do we add an Entropy Bonus to the total loss function?
   a) To minimize the MSE error of the Critic.
   b) To encourage exploration by penalizing the network if it becomes too certain (deterministic) too early.
   c) To clip the gradients.
   d) To reduce the memory usage of the Replay Buffer.

4. If an action has a negative Advantage ($A_t < 0$), and the probability ratio $r_t$ drops to $0.5$ (with $\epsilon = 0.2$), what happens to the gradient during PPO's update?
   a) The gradient becomes extremely large to quickly fix the bad action.
   b) The gradient step reverses direction.
   c) The clipping function activates ($r_t < 0.8$), and the gradient becomes $0$, stopping the update.
   d) The old policy is immediately discarded.
