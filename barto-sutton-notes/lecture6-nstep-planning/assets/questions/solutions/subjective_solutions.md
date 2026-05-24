---
layout: post
---

# Subjective Question Solutions - Lecture 5

1. **The Spectrum of RL:** Explain how n-step methods unify Monte Carlo and Temporal-Difference methods. Draw or describe the backup diagrams for $n=1$, $n=3$, and $n=\infty$.

   **Answer:**
   - **Explanation:** n-step methods sit on a spectrum between TD(0) ($n=1$) and Monte Carlo ($n=\infty$). TD(0) bootstraps after one step, making it low variance but potentially biased. MC waits for the entire episode, making it unbiased but high variance. Intermediate $n$ balances these trade-offs.
   - **Backup Diagrams:**
     - $n=1$: State -> Reward -> Next State (bootstrap).
     - $n=3$: State -> R1 -> S1 -> R2 -> S2 -> R3 -> S3 (bootstrap).
     - $n=\infty$: State -> R1 -> S1 -> ... -> RT (Terminal).

2. **Error Reduction Property:** State and explain the "error reduction property" of n-step returns. Why is this property fundamental to the convergence of n-step TD?

   **Answer:**
   - **Statement:** The worst-error of the n-step return $G_{t:t+n}$ as an estimate of $v_\pi$ is guaranteed to be less than or equal to $\gamma^n$ times the worst-error of $V_{t+n-1}$.
   - **Explanation:** This means that as $n$ increases, the "target" of our update becomes more accurate. This contraction property ensures that n-step TD converges to the true value function $v_\pi$.

3. **Off-policy n-step:** Compare and contrast off-policy n-step Sarsa (using importance sampling) with the n-step Tree Backup algorithm. What are the pros and cons of each?

   **Answer:**
   - **n-step Sarsa (IS):** Uses importance sampling ratios to weight the return. Simple but can have extremely high variance if the behavior policy $b$ is very different from target $\pi$.
   - **Tree Backup:** Does not use IS. Instead, it takes expectations over all non-taken actions at each step. It is more stable (lower variance) but can be biased if the value estimates are poor initially.

4. **The n-step Q(σ) Algorithm:** Describe the intuition behind the $n$-step $Q(\sigma)$ algorithm. How does the parameter $\sigma$ allow us to balance between sampling and expectation?

   **Answer:**
   - **Intuition:** It generalizes all n-step methods. At each step $k$, we choose $\sigma_k \in [0, 1]$.
   - If $\sigma_k = 1$, we use a sample (like Sarsa).
   - If $\sigma_k = 0$, we use an expectation (like Expected Sarsa/Tree Backup).
   - This allows dynamic adjustment of the sampling vs. expectation trade-off during an episode.

5. **Integrated Architectures:** Explain the Dyna-Q architecture. How does it handle real experience differently than simulated experience? Use the loop of *Experience -> Model -> Planning -> Value Function* in your explanation.

   **Answer:**
   - Dyna-Q integrates acting, learning, and planning. 
   - **Direct RL:** Real experience $(S, A, R, S')$ is used to update $Q$ using Q-learning.
   - **Model Learning:** Real experience is used to build a model $Model(S, A) \to (R, S')$.
   - **Planning:** The agent samples "imaginary" $(S, A)$ from the model, gets simulated $(R, S')$, and updates $Q$ again. This speeds up learning by performing many updates per real step.

6. **Dyna-Q vs. Dyna-Q+:** In a non-stationary environment, standard Dyna-Q might fail to find a new optimal path. Explain how Dyna-Q+ uses an exploration bonus to resolve this.

   **Answer:**
   - In non-stationary environments, a model might say a path is blocked when it is now open (Shortcut Maze). Standard Dyna-Q stays greedy to the old model.
   - **Dyna-Q+** adds a reward bonus $r + \kappa\sqrt{\tau}$ during planning. If a state-action hasn't been tried for a long time ($\tau$ is large), the bonus makes it attractive in "imagination," forcing the agent to eventually try it in reality and discover the change.

7. **Prioritized Sweeping:** Why is uniform sampling in the planning phase of Dyna-Q inefficient for large state spaces? How does Prioritized Sweeping use the magnitude of TD error to improve efficiency?

   **Answer:**
   - Uniform sampling in Dyna-Q is wasteful because it spends time updating states whose values wouldn't change.
   - **Prioritized Sweeping** maintains a queue of $(s, a)$ pairs. When a state $s'$ is updated and its value changes significantly, the agent looks at all predecessors $(s, a)$ that lead to $s'$ and adds them to the queue with priority equal to the magnitude of the potential change. This focuses computation on the "frontier" of learning.

8. **Expected vs. Sample Updates:** Discuss the trade-off between expected updates (DP-style) and sample updates (TD-style). In what kind of environments (in terms of branching factor $b$) is one preferred over the other?

   **Answer:**
   - **Expected updates** (DP) are exact for a given model but require $O(b)$ computation, where $b$ is the branching factor (number of next states).
   - **Sample updates** (TD) require only $O(1)$ computation.
   - As $b$ increases, sample updates become much more efficient because you can perform $b$ sample updates for the cost of one expected update, often reaching a good approximation faster.

9. **Trajectory Sampling:** Explain why focusing planning updates on states reachable under the current policy (trajectory sampling) is often more effective than uniform sweeps, especially in tasks with a large but sparse state space.

   **Answer:**
   - Uniform sweeps update the entire state space. In many tasks, most states are unreachable or irrelevant.
   - **Trajectory Sampling** simulates trajectories following the current policy. This ensures that planning effort is concentrated on the states the agent is actually likely to encounter, which is much more efficient for large, sparse state spaces.

10. **Monte Carlo Tree Search (MCTS):** Describe the four phases of MCTS (Selection, Expansion, Simulation, Backup). Why is MCTS considered a form of decision-time planning rather than background planning?

    **Answer:**
    - **Selection:** Use a tree policy (e.g., UCT) to navigate to a leaf.
    - **Expansion:** Add new nodes to the tree.
    - **Simulation:** Perform a random rollout to get a value estimate.
    - **Backup:** Update the values of all nodes in the selection path.
    - It is **decision-time** planning because the tree is built specifically for the *current state* to choose the *next action*, and the tree can be discarded or pruned after the move.
