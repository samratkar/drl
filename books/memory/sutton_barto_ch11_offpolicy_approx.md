---
name: Sutton & Barto Ch11 - Off-policy Methods with Approximation
description: Chapter 11 summary — the deadly triad (function approx + bootstrapping + off-policy), Baird's counterexample, divergence, semi-gradient off-policy methods
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 11: *Off-policy Methods with Approximation (pp. 257-286)

## Two Challenges (p. 257-258)
1. **Changing update targets**: Addressed by importance sampling (straightforward)
2. **Changing update distribution**: Not the on-policy distribution -> fundamentally problematic for stability

## Semi-gradient Off-policy Methods
- **Off-policy TD(0)**: w += alpha*rho_t*delta_t*grad v_hat(S_t,w)
- **Expected Sarsa**: w += alpha*delta_t*grad q_hat(S_t,A_t,w) — no IS needed (sums over all actions weighted by pi)
- **n-step methods**: IS ratios multiply the update; tree backup avoids IS

## The Deadly Triad (p. 264)
Instability/divergence arises when ALL THREE combine:
1. **Function approximation** (generalizing beyond tabular)
2. **Bootstrapping** (targets include estimates, not just actual returns)
3. **Off-policy training** (update distribution != on-policy distribution)

Any two are safe. The danger is NOT due to control, learning, or uncertainty — it happens even in DP.

## Counterexamples

### Baird's Counterexample (p. 261-262)
7-state MDP, canonical divergence example. All rewards zero, v_pi = 0 exactly representable. Despite this, semi-gradient TD(0) diverges to infinity. Even semi-gradient DP (expected updates, no randomness) diverges. Fixed by using on-policy distribution.

### Tsitsiklis & Van Roy (p. 263)
Even best possible least-squares fit at each iteration can diverge: w_k = ((6-4eps)/5 * gamma)^k * w_0.

### Simple w-to-2w example (p. 260)
Two states with features 1 and 2. Diverges when gamma > 0.5 under off-policy updating.

## What Can Be Given Up? (pp. 264-265)
- **Function approximation**: Can't — needed for large problems
- **Bootstrapping**: Can (use MC) but costly — delays learning, high variance with IS
- **Off-policy**: Hardest — even epsilon-greedy is mildly off-policy. Q-learning with epsilon-greedy has never been found to diverge in practice (no theoretical guarantee though)

## Averagers (p. 264)
Methods that interpolate (nearest neighbor, locally weighted regression) are guaranteed stable even with deadly triad. But popular methods (tile coding, ANNs) are NOT averagers.

## Two Approaches to the Distribution Problem
1. Importance sampling on the update distribution (warp back to on-policy)
2. True gradient methods that don't rely on any special distribution for stability
