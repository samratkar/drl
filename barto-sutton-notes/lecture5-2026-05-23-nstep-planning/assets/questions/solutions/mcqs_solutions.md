---
layout: post
---

# MCQ Solutions - Lecture 5

1. Which value of $n$ makes n-step TD prediction equivalent to the Monte Carlo method (for an episode of length $T$)?
   A. $n = 1$
   B. $n = 0$
   C. $n \ge T - t$
   D. $n = \gamma$

   **Answer: C.** $n \ge T - t$. When $n$ is large enough to reach the end of the episode, the bootstrap term $\gamma^n V(S_{t+n})$ vanishes (as $S_T$ is terminal), leaving only the discounted rewards, which is the Monte Carlo return.

2. The n-step return $G_{t:t+n}$ uses the estimated value of the state at time:
   A. $t$
   B. $t+1$
   C. $t+n$
   D. $T$ (End of episode)

   **Answer: C.** $t+n$. n-step TD waits $n$ steps to observe rewards $R_{t+1} \dots R_{t+n}$ and then uses the estimate of the next state $S_{t+n}$ to bootstrap.

3. In n-step Sarsa, the update is performed for the state-action pair $(S_t, A_t)$ at time:
   A. $t$
   B. $t+1$
   C. $t+n$
   D. $T$

   **Answer: C.** $t+n$. Updates are delayed by $n$ steps so that the $n$-step return can be computed.

4. What is the primary advantage of n-step methods over 1-step TD methods?
   A. They require no memory of past states.
   B. They can propagate reward information back multiple steps in a single update.
   C. They eliminate the need for a discount factor $\gamma$.
   D. They are always more computationally efficient per step.

   **Answer: B.** They can propagate reward information back multiple steps. 1-step TD only updates the immediately preceding state.

5. The n-step Tree Backup algorithm is designed for which type of learning?
   A. On-policy learning without bootstrapping.
   B. Off-policy learning without importance sampling.
   C. Model-based planning only.
   D. Episodic tasks with $n=1$ only.

   **Answer: B.** Off-policy learning without importance sampling. It uses expectations over all actions at each step.

6. In the $n$-step $Q(\sigma)$ algorithm, $\sigma = 1$ corresponds to:
   A. A full expectation (as in Expected Sarsa).
   B. A full sample (as in Sarsa).
   C. No update.
   D. Monte Carlo return.

   **Answer: B.** A full sample. $\sigma$ balances between sampling ($\sigma=1$) and expectation ($\sigma=0$).

7. In the Dyna architecture, the "Model" is used to:
   A. Interact directly with the real environment.
   B. Generate simulated experience for planning.
   C. Store the true transition probabilities of the world.
   D. Decrease the learning rate $\alpha$.

   **Answer: B.** Generate simulated experience. The model is a "learned" simulator.

8. Which component of Dyna-Q is responsible for improving the value function using real experience?
   A. Planning
   B. Model Learning
   C. Direct RL
   D. Search Control

   **Answer: C.** Direct RL. Direct RL updates $Q$ from real transitions; Planning updates $Q$ from model transitions.

9. Dyna-Q+ addresses non-stationary environments by adding an exploration bonus based on:
   A. The magnitude of the TD error.
   B. The number of times a state has been visited.
   C. The time elapsed since a state-action pair was last tried.
   D. The total cumulative reward.

   **Answer: C.** Time elapsed since last trial. The bonus $\kappa\sqrt{\tau}$ encourages visiting "forgotten" state-action pairs.

10. Prioritized Sweeping improves upon uniform Dyna-Q by:
    A. Updating all states in every step.
    B. Focusing planning updates on state-action pairs with large changes in their values.
    C. Eliminating the need for a model.
    D. Using only real experience for updates.

    **Answer: B.** Focusing on large changes. It uses a priority queue based on the magnitude of the TD error.

11. "Trajectory Sampling" refers to a planning strategy that:
    A. Samples state-action pairs uniformly from the model.
    B. Follows the current policy to sample transitions from the model.
    C. Only updates states that have never been visited.
    D. Uses Monte Carlo returns for all updates.

    **Answer: B.** Follows the current policy. This focuses planning on the most relevant (likely to be visited) parts of the state space.

12. As the branching factor $b$ of an environment increases, which type of update becomes relatively more efficient?
    A. Expected Updates (DP)
    B. Sample Updates (TD)
    C. Both are equally efficient.
    D. Neither is efficient.

    **Answer: B.** Sample Updates (TD). As $b$ increases, computing the full expectation (Expected update) becomes much more expensive than taking a single sample.

13. Real-time Dynamic Programming (RTDP) is a form of:
    A. Off-policy Monte Carlo.
    B. On-policy trajectory-sampling Value Iteration.
    C. Exhaustive search in the state space.
    D. Model-free Q-learning.

    **Answer: B.** On-policy trajectory-sampling Value Iteration.

14. Monte Carlo Tree Search (MCTS) is typically used for:
    A. Background planning in simple gridworlds.
    B. Decision-time planning in complex games.
    C. Estimating the model dynamics $p(s', r \mid s, a)$.
    D. Eliminating bootstrapping from TD learning.

    **Answer: B.** Decision-time planning in complex games. MCTS is performed when an action needs to be chosen for the current state.

15. In MCTS, the "Simulation" step involves:
    A. Updating the tree with exact Bellman equations.
    B. Running a complete episode (rollout) using a default policy.
    C. Expanding all possible child nodes of the current leaf.
    D. Selecting the best action using a greedy policy.

    **Answer: B.** Running a complete episode (rollout) using a default policy. This provides a sample return to estimate the value of a leaf node.
