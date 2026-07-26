---
layout: post
---

# Programming Questions (Lecture 12: Implementation Frameworks)

1. **Implementing a Gymnasium Evaluation Loop:**
   Write a complete Python evaluation function `evaluate_policy` using Gymnasium. The function must run the policy for $N$ episodes, select actions by calling `policy_fn(state)`, sum the total rewards, and return the average reward per episode.
   Ensure that you correctly reset the environment with a seed, handle the `terminated` and `truncated` flags to end episodes, and close the environment at the end.

   ```python
   import gymnasium as gym

   def evaluate_policy(env_id, policy_fn, num_episodes=10, seed=42):
       """
       Args:
           env_id: str, e.g., "CartPole-v1"
           policy_fn: callable, takes state and returns action
           num_episodes: int, number of episodes to run
           seed: int, random seed
       Returns:
           avg_reward: float, average reward accumulated per episode
       """
       # TODO: Implement the evaluation loop
       pass
   ```
