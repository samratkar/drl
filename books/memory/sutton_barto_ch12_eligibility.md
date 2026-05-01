---
name: Sutton & Barto Ch12 - Eligibility Traces
description: Chapter 12 summary — lambda-return, TD(lambda), forward/backward views, accumulating/dutch/replacing traces, true online TD(lambda), Sarsa(lambda), off-policy traces (GTD/GQ/HTD/Emphatic)
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 12: Eligibility Traces (pp. 287-320)

## Core Idea
Eligibility traces z_t in R^d parallel the weight vector w_t. When a component participates in an estimate, its trace is bumped up then fades at rate gamma*lambda. Learning occurs if a nonzero TD error arrives before the trace fades.

## lambda-Return (p. 289)
G_t^lambda = (1-lambda) sum_{n=1}^{T-t-1} lambda^{n-1} G_{t:t+n} + lambda^{T-t-1} G_t
Weighted average of all n-step returns. lambda=0 -> one-step TD. lambda=1 -> MC.

## Forward vs Backward View (pp. 288-294)
- **Forward**: Look ahead to compute lambda-return (theoretical ideal, not practical)
- **Backward**: Current TD error propagated back through eligibility traces (practical implementation)
- Exactly equivalent for true online TD(lambda) with linear function approximation.

## Algorithms

### Semi-gradient TD(lambda) (p. 293)
z <- gamma*lambda*z + grad v_hat(S,w)  [accumulating trace]
delta <- R + gamma*v_hat(S',w) - v_hat(S,w)
w <- w + alpha*delta*z

### True Online TD(lambda) (p. 300)
z <- gamma*lambda*z + (1 - alpha*gamma*lambda*z^T*x)*x  [dutch trace]
w <- w + alpha*(delta + V - V_old)*z - alpha*(V - V_old)*x
Produces exactly the same weight sequence as the online lambda-return algorithm. ~50% more computation per step than standard TD(lambda). Same O(d) memory.

### Sarsa(lambda) (p. 305)
Action-value version. z <- gamma*lambda*z + grad q_hat(S,A,w). True online Sarsa(lambda) performs best.

## Trace Types (p. 301)
- **Accumulating**: z_t = gamma*lambda*z_{t-1} + grad v_hat(S_t,w). Standard.
- **Dutch**: z_t = gamma*lambda*z_{t-1} + (1-alpha*gamma*lambda*z^T*x)*x. Used in true online methods. Generally best.
- **Replacing**: z_i = 1 if x_i=1, else gamma*lambda*z_i. Crude approximation to dutch. Only for binary features.

## Error Bound (p. 295)
VE(w_inf) <= (1-gamma*lambda)/(1-gamma) * min_w VE(w)
As lambda->1, bound approaches minimum. lambda=0 is loosest. But in practice, intermediate lambda usually best.

## Off-policy with Traces
All involve the deadly triad when lambda < 1. Stable algorithms for linear function approximation:
- **GTD(lambda)** (p. 314): Extension of TDC with traces. Two weight vectors w, v.
- **GQ(lambda)** (p. 315): Gradient-TD for action values.
- **HTD(lambda)** (p. 315): Hybrid — reduces to TD(lambda) when on-policy (GTD does not).
- **Emphatic TD(lambda)** (p. 315): Strong convergence guarantees but high variance.
- **TB(lambda)** (p. 313): Tree-Backup with traces. No IS needed. z_t = gamma*lambda*pi(A_t|S_t)*z_{t-1} + grad q_hat.

## Key Example
**Mountain Car** (p. 305-306): True online Sarsa(lambda) outperforms n-step Sarsa, Sarsa(lambda) with accumulating/replacing traces. Intermediate lambda is best across four benchmark tasks (Figure 12.14).
