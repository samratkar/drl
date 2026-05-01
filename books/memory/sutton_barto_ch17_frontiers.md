---
name: Sutton & Barto Ch17 - Frontiers
description: Chapter 17 summary — general value functions (GVFs), options/temporal abstraction, POMDPs, PSRs, observations vs state, intrinsic motivation, reward design, open challenges
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 17: Frontiers (pp. 459-478)

## General Value Functions (GVFs) (pp. 459-461)
Generalize value functions to predict arbitrary signals (not just reward). A GVF predicts expected discounted sum of a **cumulant signal** C_t under policy pi with termination gamma.
v_{pi,gamma,C}(s) = E[sum (prod gamma(S_i)) * C_{k+1} | S_t=s]

**Auxiliary tasks**: Additional prediction/control tasks sharing representations with the main reward task. Enable "Pavlovian control" (connect predictions to reflexive actions).

## Options — Temporal Abstraction (pp. 461-463)
Temporally extended courses of action. Option omega = (pi_omega, gamma_omega): a policy and termination function.
- **Option models**: Reward part r(s,omega) and transition part p(s'|s,omega)
- **Hierarchical policies**: Select from options, not just primitive actions
- Bellman equation: v_pi(s) = sum_omega pi(omega|s)[r(s,omega) + sum_{s'} p(s'|s,omega)*v_pi(s')]
- Primitive actions are special cases (gamma_omega(s)=0 everywhere)

## Observations and State (pp. 464-468)
- **Markov state**: Compact summary S_t=f(H_t) satisfying Markov property for all future tests
- **State-update function**: S_{t+1} = u(S_t, A_t, O_{t+1}) — must be efficiently computable
- **POMDPs**: Natural Markov state = belief state (distribution over latent states). Exact solutions scale poorly.
- **PSRs** (Predictive State Representations): State = vector of probabilities of "core tests." Grounded in observables.
- **k-th order history**: S_t = (O_t, A_{t-1}, ..., A_{t-k}). Simple but effective. DQN's 4-frame stacking is an example.

## Reward Design (pp. 469-471)
- **Shaping**: Modify reward to guide learning from simple to target behavior
- **Prior knowledge**: v_hat(s,w) = w^T x(s) + v_0(s)
- **Inverse RL**: Recover reward from expert behavior. Not uniquely determined.

## Six Open Challenges (pp. 472-474)
1. **Incremental function approximation**: Deep learning struggles with online/incremental settings (catastrophic interference)
2. **Representation learning / meta-learning**: Learning features that make future learning faster
3. **Scalable planning with learned models**: Most model-based RL still uses tabular models
4. **Automated task selection**: Agents choose their own GVFs, options, auxiliary tasks
5. **Computational curiosity / intrinsic motivation**: Internal reward (novelty, learning progress) to drive exploration
6. **Safe real-world embedding**: Deploying RL agents in physical environments safely

## Future of AI (pp. 475-478)
- Deep RL behind remarkable AI achievements
- Gap between artificial and human intelligence remains great (general adaptability, few-shot learning)
- Safety: reward design problems ("Sorcerer's Apprentice"), risk management
- RL contributions: understanding the brain, advising human decision-making, advancing AI
