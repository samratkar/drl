---
name: Sutton & Barto Ch2 - Multi-Armed Bandits
description: Chapter 2 summary — k-armed bandits, epsilon-greedy, incremental updates, UCB, gradient bandits, nonstationary tracking, optimistic initialization
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 2: Multi-Armed Bandits (pp. 25-45)

## Problem Setup
Choose among k actions repeatedly; receive reward from stationary distribution depending on action. Goal: maximize expected total reward. No state — single situation (nonassociative).

## Key Equations
- **True value**: q_*(a) = E[R_t | A_t = a]
- **Sample-average**: Q_t(a) = sum(rewards when a taken) / count(a taken)
- **Incremental update**: Q_{n+1} = Q_n + (1/n)[R_n - Q_n]. General form: NewEstimate <- OldEstimate + StepSize[Target - OldEstimate]
- **Exponential recency-weighted**: Q_{n+1} = Q_n + alpha[R_n - Q_n] = (1-alpha)^n Q_1 + sum alpha(1-alpha)^{n-i} R_i
- **Convergence conditions**: sum alpha_n = inf (overcome initial conditions), sum alpha_n^2 < inf (steps shrink). 1/n satisfies both; constant alpha does not (desirable for nonstationary).

## Methods
- **Epsilon-greedy**: Greedy most of time, random action with prob epsilon. Beats pure greedy when reward variance > 0.
- **Optimistic initial values** (p. 34): Set Q_1 high (e.g., +5); greedy selection causes systematic early exploration. Only helps initially, not for nonstationary.
- **UCB**: A_t = argmax_a [Q_t(a) + c*sqrt(ln(t)/N_t(a))]. Selects non-greedy actions by uncertainty, not indiscriminately. Generally best on stationary bandits but hard to extend to general RL.
- **Gradient bandits**: Learn preferences H_t(a), select via softmax pi_t(a) = exp(H_t(a))/sum. Update: H_{t+1}(A_t) = H_t(A_t) + alpha(R_t - R_bar)(1 - pi_t(A_t)); H_{t+1}(a) = H_t(a) - alpha(R_t - R_bar)pi_t(a) for a != A_t. Proved to be stochastic gradient ascent on E[R_t]. Baseline R_bar is critical for variance reduction.

## 10-Armed Testbed (p. 28)
2000 problems, q_*(a) ~ N(0,1), rewards ~ N(q_*(a),1). Epsilon=0.1 and 0.01 both outperform greedy long-term. Greedy gets stuck ~1.0 reward (optimal ~1.55), finds best action only ~1/3 of time.

## Key Distinctions
- **Evaluative vs instructive feedback**: RL uses evaluative (how good was this action) not instructive (what was the correct action)
- **Associative search / contextual bandits** (p. 41): Different situations arise, must learn situation->action mapping, but actions affect only immediate reward. Intermediate between bandits and full RL.
