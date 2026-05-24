---
layout: post
---

# Subjective Questions - MDPs & Dynamic Programming

1. **Agent-Environment Boundary**: Explain the concept of the boundary between the agent and the environment. Provide an example where something typically considered part of the agent's body is actually part of the environment.

2. **Markov Property**: Formally define the Markov property. Why is this property fundamental to the formulation of Reinforcement Learning problems as MDPs?

3. **Dynamics Function**: Describe the dynamics function $p(s', r \mid s, a)$. How does it encapsulate all the physics/rules of the environment?

4. **Returns & Discounting**: Define the return $G_t$ for a continuing task. Explain how the discount factor $\gamma$ prevents the return from becoming infinite and how it adjusts the agent's time horizon.

5. **State-Value vs. Action-Value**: Contrast $v_\pi(s)$ and $q_\pi(s,a)$. Why might we prefer one over the other in different algorithms?

6. **Bellman Equation Intuition**: Explain the intuition behind the Bellman equation for $v_\pi(s)$. How does it represent the idea of "recursive consistency"?

7. **Optimal Value Functions**: Define $v^*(s)$ and $q^*(s,a)$. What is the relationship between these two optimal functions?

8. **Bellman Optimality Equation**: Write the Bellman optimality equation for $v^*(s)$. Why is this equation non-linear compared to the Bellman equation for a fixed policy?

9. **Policy Improvement Theorem**: State the Policy Improvement Theorem. What does it guarantee about the relationship between a policy $\pi$ and its greedy improvement $\pi'$?

10. **Policy Iteration Workflow**: Describe the three main steps of the Policy Iteration algorithm: Initialization, Policy Evaluation, and Policy Improvement.

11. **Value Iteration Mechanics**: How does Value Iteration combine evaluation and improvement into a single update? How is it different from Policy Iteration in terms of computational steps?

12. **Generalized Policy Iteration (GPI)**: Explain the concept of GPI. Use the analogy of two competing/cooperating processes (evaluation and improvement) to describe how they lead to the optimal solution.

13. **Asynchronous DP**: What are asynchronous DP algorithms? Why are they useful for problems with very large state spaces?

14. **Bootstrapping**: Define "bootstrapping" in the context of DP. Give an example of an update rule that involves bootstrapping.

15. **Jack's Car Rental**: Describe the Jack's Car Rental problem. What are the state, actions, and rewards? How does the Poisson distribution play a role in the dynamics?

16. **Gambler's Problem**: In the Gambler's Problem, explain why the reward is 0 for all transitions except those that reach the goal capital (e.g., 100). How does this affect the value function?

17. **Gridworld Boundary Conditions**: In a typical gridworld, how are "off-the-grid" actions handled in terms of state transitions and rewards to ensure the environment remains well-defined?

18. **Unifying Episodic and Continuing Tasks**: Explain the "absorbing state" notation used to unify episodic and continuing tasks into a single mathematical framework.

19. **Policy vs. Value**: Can an optimal policy be non-greedy with respect to its own value function? Explain why or why not using the Bellman Optimality Equation.

20. **Complexity of DP**: Discuss the computational complexity of DP. Why is it said to suffer from the "curse of dimensionality"?
