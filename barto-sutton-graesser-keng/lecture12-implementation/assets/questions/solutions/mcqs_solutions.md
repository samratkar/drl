---
layout: post
---

# MCQ Solutions (Lecture 12: Implementation Frameworks)

1. **Answer: b**
   * **Explanation:** `terminated` indicates that the agent has reached an actual terminal state defined by the environment task (e.g. falling into a pit, crashing, or winning). `truncated` indicates that the episode was cut off prematurely by an external limit (e.g. hitting the step limit of 200).

2. **Answer: b**
   * **Explanation:** `Box` represents a continuous multidimensional interval. `Discrete` represents integer values. `Tuple` and `Dict` are containers.

3. **Answer: a**
   * **Explanation:** A **Trial** in SLM Lab runs multiple sessions (typically 4 or more) using different random seeds for the same configuration, allowing researchers to calculate standard deviation and statistical averages.

4. **Answer: c**
   * **Explanation:** Actor-Critic based methods (including n-step A2C) are implemented in `slm_lab/agent/algorithm/actor_critic.py`. `reinforce.py` contains REINFORCE, and `ppo.py` contains PPO.
