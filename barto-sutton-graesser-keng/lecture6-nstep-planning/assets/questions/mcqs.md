---
layout: post
---

# Multiple Choice Questions - Lecture 5 (n-step Bootstrapping & Planning)

1. Which value of $n$ makes n-step TD prediction equivalent to the Monte Carlo method (for an episode of length $T$)?
   A. $n = 1$
   B. $n = 0$
   C. $n \ge T - t$
   D. $n = \gamma$

2. The n-step return $G_{t:t+n}$ uses the estimated value of the state at time:
   A. $t$
   B. $t+1$
   C. $t+n$
   D. $T$ (End of episode)

3. In n-step Sarsa, the update is performed for the state-action pair $(S_t, A_t)$ at time:
   A. $t$
   B. $t+1$
   C. $t+n$
   D. $T$

4. What is the primary advantage of n-step methods over 1-step TD methods?
   A. They require no memory of past states.
   B. They can propagate reward information back multiple steps in a single update.
   C. They eliminate the need for a discount factor $\gamma$.
   D. They are always more computationally efficient per step.

5. The n-step Tree Backup algorithm is designed for which type of learning?
   A. On-policy learning without bootstrapping.
   B. Off-policy learning without importance sampling.
   |C. Model-based planning only.
   D. Episodic tasks with $n=1$ only.

6. In the $n$-step $Q(\sigma)$ algorithm, $\sigma = 1$ corresponds to:
   A. A full expectation (as in Expected Sarsa).
   B. A full sample (as in Sarsa).
   C. No update.
   D. Monte Carlo return.

7. In the Dyna architecture, the "Model" is used to:
   A. Interact directly with the real environment.
   B. Generate simulated experience for planning.
   C. Store the true transition probabilities of the world.
   D. Decrease the learning rate $\alpha$.

8. Which component of Dyna-Q is responsible for improving the value function using real experience?
   A. Planning
   B. Model Learning
   C. Direct RL
   D. Search Control

9. Dyna-Q+ addresses non-stationary environments by adding an exploration bonus based on:
   A. The magnitude of the TD error.
   B. The number of times a state has been visited.
   C. The time elapsed since a state-action pair was last tried.
   D. The total cumulative reward.

10. Prioritized Sweeping improves upon uniform Dyna-Q by:
    A. Updating all states in every step.
    B. Focusing planning updates on state-action pairs with large changes in their values.
    C. Eliminating the need for a model.
    D. Using only real experience for updates.

11. "Trajectory Sampling" refers to a planning strategy that:
    A. Samples state-action pairs uniformly from the model.
    B. Follows the current policy to sample transitions from the model.
    C. Only updates states that have never been visited.
    D. Uses Monte Carlo returns for all updates.

12. As the branching factor $b$ of an environment increases, which type of update becomes relatively more efficient?
    A. Expected Updates (DP)
    B. Sample Updates (TD)
    C. Both are equally efficient.
    D. Neither is efficient.

13. Real-time Dynamic Programming (RTDP) is a form of:
    A. Off-policy Monte Carlo.
    B. On-policy trajectory-sampling Value Iteration.
    C. Exhaustive search in the state space.
    D. Model-free Q-learning.

14. Monte Carlo Tree Search (MCTS) is typically used for:
    A. Background planning in simple gridworlds.
    B. Decision-time planning in complex games.
    C. Estimating the model dynamics $p(s', r \mid s, a)$.
    D. Eliminating bootstrapping from TD learning.

15. In MCTS, the "Simulation" step involves:
    A. Updating the tree with exact Bellman equations.
    B. Running a complete episode (rollout) using a default policy.
    C. Expanding all possible child nodes of the current leaf.
    D. Selecting the best action using a greedy policy.
