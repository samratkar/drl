---
name: Sutton & Barto Ch7 - n-step Bootstrapping
description: Chapter 7 summary — n-step TD/Sarsa, n-step returns, importance sampling for n-step, tree backup algorithm, Q(sigma) unifying algorithm
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 7: n-step Bootstrapping (pp. 141-158)

## Core Idea
Unifies MC (full return) and one-step TD. The n-step return uses n actual rewards plus a bootstrapped estimate n steps later. Intermediate n values often outperform both extremes.

## n-step Return (p. 143)
G_{t:t+n} = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n} + gamma^n * V_{t+n-1}(S_{t+n})
If t+n >= T, reduces to the full return G_t.

## Algorithms

### n-step TD (p. 144)
V_{t+n}(S_t) = V_{t+n-1}(S_t) + alpha[G_{t:t+n} - V_{t+n-1}(S_t)]
**Error reduction**: max_s |E[G_{t:t+n}|S_t=s] - v_pi(s)| <= gamma^n * max_s |V_{t+n-1}(s) - v_pi(s)|

### n-step Sarsa (p. 146)
Same structure with action values. G_{t:t+n} = ... + gamma^n * Q_{t+n-1}(S_{t+n}, A_{t+n}).
After single episode, 10-step Sarsa strengthens last 10 actions (vs just 1 for one-step Sarsa).

### Off-policy n-step Sarsa (p. 149)
Q_{t+n}(S_t,A_t) += alpha * rho_{t+1:t+n} * [G_{t:t+n} - Q_{t+n-1}(S_t,A_t)]
Note: IS ratio starts at t+1 (not t) because we're updating a state-action pair.

### n-step Tree Backup (p. 152-154)
Off-policy without importance sampling. Weights unselected actions by target-policy probabilities. Recursive: G_{t:t+n} = R_{t+1} + gamma*sum_{a!=A_{t+1}} pi(a|S_{t+1})*Q(S_{t+1},a) + gamma*pi(A_{t+1}|S_{t+1})*G_{t+1:t+n}

### Q(sigma) — Unifying Algorithm (p. 155)
sigma_t in [0,1] controls degree of sampling per step. sigma=1 -> Sarsa (sample). sigma=0 -> tree backup (expectation). Intermediate values blend both approaches.

## Control Variates (p. 150)
Modified off-policy return: G_{t:h} = rho_t(R_{t+1} + gamma*G_{t+1:h}) + (1-rho_t)*V_{h-1}(S_t). When rho=0, target equals current estimate (no change) rather than zeroing out. Reduces variance.

## Key Example
**19-state Random Walk** (p. 144-145): Intermediate n (4 or 8) consistently best. Demonstrates practical value of the n-step spectrum.

## Key Distinctions
- **n-step Sarsa vs Tree Backup vs Expected Sarsa**: Sarsa = all sample, needs IS for off-policy. Tree backup = all branched, no IS needed. Expected Sarsa = sample except last step (branched).
- **IS vs tree backup for off-policy**: IS is simple but high variance. Tree backup avoids IS but may effectively span fewer steps when target and behavior differ greatly.
