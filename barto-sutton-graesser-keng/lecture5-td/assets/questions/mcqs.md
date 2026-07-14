---
layout: post
---

# Multiple Choice Questions (Lecture 5: Temporal Difference Learning)

1. What is the fundamental difference between TD(0) and Monte Carlo methods?
   a) TD methods require a model of the environment; MC does not.
   b) MC updates require waiting until the end of an episode, while TD can learn before the final outcome.
   c) TD methods only work on continuous state spaces.
   d) MC methods bootstrap; TD methods do not.

2. In Q-learning, what determines the target for the action-value update?
   a) The action actually taken in the next state.
   b) The maximum Q-value over all possible actions in the next state.
   c) The average Q-value of the next state.
   d) The return $G_t$ at the end of the episode.

3. Which of the following is true about SARSA?
   a) It is an off-policy algorithm.
   b) It uses the greedy action for the target update, regardless of the action actually chosen.
   c) It is an on-policy algorithm.
   d) It always converges faster than Q-learning.

4. Why might Double Q-learning be preferred over standard Q-learning?
   a) It eliminates the need for an epsilon-greedy policy.
   b) It mitigates the maximization bias that causes Q-learning to overestimate action values.
   c) It perfectly mimics the SARSA update.
   d) It uses half the memory of standard Q-learning.
