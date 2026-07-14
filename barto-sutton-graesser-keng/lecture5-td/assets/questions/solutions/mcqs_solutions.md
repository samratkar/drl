---
layout: post
---

# Solutions to MCQs (Lecture 5)

1. **b) MC updates require waiting until the end of an episode, while TD can learn before the final outcome.** (TD methods bootstrap, allowing them to update estimates based on other learned estimates one step ahead).
2. **b) The maximum Q-value over all possible actions in the next state.** (This makes Q-learning an off-policy algorithm).
3. **c) It is an on-policy algorithm.** (SARSA updates using the action that the $\epsilon$-greedy policy actually takes in the next state).
4. **b) It mitigates the maximization bias that causes Q-learning to overestimate action values.** (By decoupling action selection from action evaluation using two separate Q-tables).
