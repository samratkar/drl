---
name: Sutton & Barto Ch3 - Finite MDPs
description: Chapter 3 summary — MDP formalization, dynamics function p(s'r|sa), Bellman equations, optimal value functions, Bellman optimality equations, v_pi vs q_pi
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 3: Finite Markov Decision Processes (pp. 47-70)

## Agent-Environment Interface (p. 47-48)
At each step t: agent in state S_t, takes action A_t, receives reward R_{t+1}, transitions to S_{t+1}.
Trajectory: S_0, A_0, R_1, S_1, A_1, R_2, ...

## Dynamics Function (p. 48)
p(s', r | s, a) = Pr{S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a}. Completely characterizes the MDP.
Derived quantities:
- p(s'|s,a) = sum_r p(s',r|s,a)
- r(s,a) = sum_r r * sum_{s'} p(s',r|s,a)
- r(s,a,s') = sum_r r * p(s',r|s,a) / p(s'|s,a)

## Returns
- **Episodic**: G_t = R_{t+1} + R_{t+2} + ... + R_T
- **Discounted**: G_t = sum_{k=0}^inf gamma^k R_{t+k+1} = R_{t+1} + gamma*G_{t+1}
- **Unified**: G_t = sum_{k=t+1}^T gamma^{k-t-1} R_k (with T=inf or gamma=1, not both)

## Value Functions (p. 58)
- v_pi(s) = E_pi[G_t | S_t=s] — state-value function
- q_pi(s,a) = E_pi[G_t | S_t=s, A_t=a] — action-value function

## Bellman Equation (p. 59) — THE CENTRAL EQUATION
v_pi(s) = sum_a pi(a|s) sum_{s',r} p(s',r|s,a)[r + gamma*v_pi(s')]
Unique solution. System of |S| linear equations in |S| unknowns.

## Optimal Value Functions (pp. 62-65)
- v_*(s) = max_pi v_pi(s) = max_a sum_{s',r} p(s',r|s,a)[r + gamma*v_*(s')]
- q_*(s,a) = sum_{s',r} p(s',r|s,a)[r + gamma*max_{a'} q_*(s',a')]
- q_*(s,a) = E[R_{t+1} + gamma*v_*(S_{t+1}) | S_t=s, A_t=a]

## Key Properties
- **Partial ordering**: pi >= pi' iff v_pi(s) >= v_{pi'}(s) for all s. At least one optimal policy exists.
- **v_* -> optimal policy**: Greedy w.r.t. v_* is optimal, but requires knowing dynamics (one-step lookahead).
- **q_* -> optimal policy**: Simply pick argmax_a q_*(s,a). No model needed.
- **Reward hypothesis** (p. 53): All goals can be expressed as maximizing cumulative scalar reward.
- **Agent-environment boundary** (p. 50): Not the physical boundary; anything the agent cannot arbitrarily change is environment.

## Key Examples
- Gridworld (p. 60): 5x5 grid with special teleporting states A->A' (+10) and B->B' (+5)
- Recycling Robot (p. 52): S={high,low}, demonstrates complete MDP specification
- Golf (p. 61): Illustrates v_pi vs q_pi difference
