---
layout: post
---

# Multiple Choice Questions (Lecture 12: Implementation Frameworks)

1. In Gymnasium, what is the difference between the `terminated` and `truncated` flags returned by `env.step(action)`?
   a) `terminated` indicates the agent crashed; `truncated` indicates the agent won.
   b) `terminated` indicates reaching a natural end of the episode (terminal state); `truncated` indicates ending due to an external limit (e.g. time limit).
   c) `terminated` is for continuous environments; `truncated` is for discrete environments.
   d) `terminated` stops the environment loop; `truncated` pauses it.

2. Which space type in Gymnasium represents a bounded continuous interval of float values (e.g., box state coordinates or action ranges)?
   a) `Discrete`
   b) `Box`
   c) `Tuple`
   d) `Dict`

3. In SLM Lab's hierarchy, which of the following represents a run of the same algorithm config using multiple sessions with different random seeds?
   a) Trial
   b) Experiment
   c) Session
   d) Epoch

4. In the Laura Graesser book (and SLM Lab), where is the Advantage Actor-Critic (A2C) agent algorithm logic implemented?
   a) `slm_lab/agent/algorithm/reinforce.py`
   b) `slm_lab/agent/algorithm/sarsa.py`
   c) `slm_lab/agent/algorithm/actor_critic.py`
   d) `slm_lab/agent/algorithm/ppo.py`
