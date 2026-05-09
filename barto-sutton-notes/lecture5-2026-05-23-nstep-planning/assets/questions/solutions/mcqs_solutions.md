---
layout: post
---

# MCQ Solutions - Lecture 5

1. **Answer: C.** $n \ge T - t$. When $n$ is large enough to reach the end of the episode, the bootstrap term $\gamma^n V(S_{t+n})$ vanishes (as $S_T$ is terminal), leaving only the discounted rewards, which is the Monte Carlo return.
2. **Answer: C.** $t+n$. n-step TD waits $n$ steps to observe rewards $R_{t+1} \dots R_{t+n}$ and then uses the estimate of the next state $S_{t+n}$ to bootstrap.
3. **Answer: C.** $t+n$. Updates are delayed by $n$ steps so that the $n$-step return can be computed.
4. **Answer: B.** They can propagate reward information back multiple steps. 1-step TD only updates the immediately preceding state.
5. **Answer: B.** Off-policy learning without importance sampling. It uses expectations over all actions at each step.
6. **Answer: B.** A full sample. $\sigma$ balances between sampling ($\sigma=1$) and expectation ($\sigma=0$).
7. **Answer: B.** Generate simulated experience. The model is a "learned" simulator.
8. **Answer: C.** Direct RL. Direct RL updates $Q$ from real transitions; Planning updates $Q$ from model transitions.
9. **Answer: C.** Time elapsed since last trial. The bonus $\kappa\sqrt{\tau}$ encourages visiting "forgotten" state-action pairs.
10. **Answer: B.** Focusing on large changes. It uses a priority queue based on the magnitude of the TD error.
11. **Answer: B.** Follows the current policy. This focuses planning on the most relevant (likely to be visited) parts of the state space.
12. **Answer: B.** Sample Updates (TD). As $b$ increases, computing the full expectation (Expected update) becomes much more expensive than taking a single sample.
13. **Answer: B.** On-policy trajectory-sampling Value Iteration.
14. **Answer: B.** Decision-time planning in complex games. MCTS is performed when an action needs to be chosen for the current state.
15. **Answer: B.** Running a complete episode (rollout) using a default policy. This provides a sample return to estimate the value of a leaf node.
