---
layout: post
---

# Subjective Questions - Lecture 5 (n-step Bootstrapping & Planning)

1. **The Spectrum of RL:** Explain how n-step methods unify Monte Carlo and Temporal-Difference methods. Draw or describe the backup diagrams for $n=1$, $n=3$, and $n=\infty$.

2. **Error Reduction Property:** State and explain the "error reduction property" of n-step returns. Why is this property fundamental to the convergence of n-step TD?

3. **Off-policy n-step:** Compare and contrast off-policy n-step Sarsa (using importance sampling) with the n-step Tree Backup algorithm. What are the pros and cons of each?

4. **The n-step Q(σ) Algorithm:** Describe the intuition behind the $n$-step $Q(\sigma)$ algorithm. How does the parameter $\sigma$ allow us to balance between sampling and expectation?

5. **Integrated Architectures:** Explain the Dyna-Q architecture. How does it handle real experience differently than simulated experience? Use the loop of *Experience -> Model -> Planning -> Value Function* in your explanation.

6. **Dyna-Q vs. Dyna-Q+:** In a non-stationary environment (like a maze where a path is suddenly blocked or opened), standard Dyna-Q might fail to find a new optimal path. Explain how Dyna-Q+ uses an exploration bonus to resolve this.

7. **Prioritized Sweeping:** Why is uniform sampling in the planning phase of Dyna-Q inefficient for large state spaces? How does Prioritized Sweeping use the magnitude of TD error to improve efficiency?

8. **Expected vs. Sample Updates:** Discuss the trade-off between expected updates (DP-style) and sample updates (TD-style). In what kind of environments (in terms of branching factor $b$) is one preferred over the other?

9. **Trajectory Sampling:** Explain why focusing planning updates on states reachable under the current policy (trajectory sampling) is often more effective than uniform sweeps, especially in tasks with a large but sparse state space.

10. **Monte Carlo Tree Search (MCTS):** Describe the four phases of MCTS (Selection, Expansion, Simulation, Backup). Why is MCTS considered a form of decision-time planning rather than background planning?
