---
name: Sutton & Barto Ch4 - Dynamic Programming
description: Chapter 4 summary — policy evaluation, policy improvement theorem, policy iteration, value iteration, GPI, async DP, efficiency of DP
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 4: Dynamic Programming (pp. 73-90)

## Core Idea
DP = compute optimal policies given a perfect model (MDP). Turn Bellman equations into update rules.

## Algorithms

### Iterative Policy Evaluation (p. 75)
Compute v_pi by repeated sweeps: V_{k+1}(s) = sum_a pi(a|s) sum_{s',r} p(s',r|s,a)[r + gamma*V_k(s')]
Stop when max|V_{k+1}(s) - V_k(s)| < theta. In-place version (Gauss-Seidel) converges faster.

### Policy Improvement (pp. 76-79)
- Compute q_pi(s,a) = sum_{s',r} p(s',r|s,a)[r + gamma*v_pi(s')]
- **Policy Improvement Theorem**: If q_pi(s, pi'(s)) >= v_pi(s) for all s, then v_{pi'} >= v_pi everywhere.
- **Greedy policy**: pi'(s) = argmax_a sum_{s',r} p(s',r|s,a)[r + gamma*v_pi(s')]

### Policy Iteration (p. 80)
Alternate: complete policy evaluation -> policy improvement. Converges in finite iterations (finite number of deterministic policies). Each step is guaranteed to improve or reach optimality.

### Value Iteration (p. 83)
V_{k+1}(s) = max_a sum_{s',r} p(s',r|s,a)[r + gamma*V_k(s')]
Truncated policy iteration with just one evaluation sweep. Equivalent to Bellman optimality equation as update rule. Extract policy at the end.

### Generalized Policy Iteration (GPI) (p. 86)
The general idea of interleaving evaluation and improvement. Almost all RL methods are GPI. Stability (no changes) = optimality.

## Asynchronous DP (p. 85)
Update states in any order without systematic sweeps. Can focus on relevant states. Enables real-time interaction.

## Key Examples
- **4x4 Gridworld** (p. 76): Greedy policies w.r.t. intermediate value estimates can already be optimal after k=3 iterations
- **Jack's Car Rental** (p. 81): Policy iteration converging pi_0 through pi_4 to optimal
- **Gambler's Problem** (p. 84): Value iteration with interesting optimal policy structure

## Properties
- DP is polynomial in |S| and |A|; exponentially faster than policy-space search (k^n policies)
- **Curse of dimensionality**: States grow exponentially with state variables — inherent to the problem, not DP
- All DP updates are **expected updates** (over all successors), distinguishing them from sample-based methods
- DP methods **bootstrap** (update estimates from estimates)
