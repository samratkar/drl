---
name: Sutton & Barto Ch6 - Temporal-Difference Learning
description: Chapter 6 summary — TD(0), TD error, Sarsa, Q-learning, Expected Sarsa, Double Q-learning, maximization bias, batch TD vs MC, certainty-equivalence
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 6: Temporal-Difference Learning (pp. 119-138)

## Core Idea
Combines MC (learn from experience, no model) with DP (bootstrap from estimates). Updates after each step, not after episode completion.

## TD(0) Prediction (p. 120)
V(S_t) <- V(S_t) + alpha[R_{t+1} + gamma*V(S_{t+1}) - V(S_t)]

**TD error**: delta_t = R_{t+1} + gamma*V(S_{t+1}) - V(S_t)
**TD target**: R_{t+1} + gamma*V(S_{t+1}) — estimate for two reasons: samples expected values AND uses estimated V.

**MC error = sum of TD errors** (p. 121): G_t - V(S_t) = sum_{k=t}^{T-1} gamma^{k-t} delta_k (if V unchanged during episode)

## TD Control Algorithms

### Sarsa — On-policy (p. 129)
Q(S_t,A_t) <- Q(S_t,A_t) + alpha[R_{t+1} + gamma*Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]
Named after (S,A,R,S',A'). Learns values for the policy being followed (including exploration).

### Q-learning — Off-policy (p. 131)
Q(S_t,A_t) <- Q(S_t,A_t) + alpha[R_{t+1} + gamma*max_a Q(S_{t+1},a) - Q(S_t,A_t)]
Directly approximates q_* regardless of behavior policy. Converges under usual conditions.

### Expected Sarsa (p. 133)
Q(S_t,A_t) <- Q(S_t,A_t) + alpha[R_{t+1} + gamma*sum_a pi(a|S_{t+1})*Q(S_{t+1},a) - Q(S_t,A_t)]
Uses expected value under current policy instead of sample. Eliminates variance from A' selection. Subsumes Q-learning (when target is greedy). Generally outperforms both Sarsa and Q-learning.

### Double Q-learning (p. 136)
Maintains Q_1 and Q_2. With prob 0.5: Q_1(S,A) <- Q_1(S,A) + alpha[R + gamma*Q_2(S', argmax_a Q_1(S',a)) - Q_1(S,A)]. Eliminates **maximization bias** — the positive bias when max over estimated values overestimates max of true values.

## Batch TD vs Batch MC (pp. 126-128)
- **Batch MC**: Converges to values minimizing mean-squared error on training data
- **Batch TD**: Converges to **certainty-equivalence estimate** — correct for the maximum-likelihood MDP model. Better for future prediction when process is Markov.

## Key Examples
- **Random Walk** (p. 125): 5 states, TD(0) consistently outperforms MC across alpha values
- **Cliff Walking** (p. 132): Q-learning learns optimal but risky path (cliff edge); Sarsa learns safer longer path. Key illustration of on-policy vs off-policy.
- **Maximization Bias** (p. 134): Q-learning takes suboptimal left action ~75%; Double Q-learning eliminates the bias.

## Convergence
- TD(0): Converges to v_pi in the mean (small constant alpha) or with probability 1 (decreasing alpha per stochastic approximation conditions)
- Sarsa: Converges with prob 1 to optimal if all pairs visited infinitely often and policy converges to greedy
- Q-learning: Converges with prob 1 to q_* under standard conditions

## Key Distinctions
- **TD vs MC**: TD bootstraps, learns step-by-step, works for continuing tasks. MC waits for episode end, no bootstrapping.
- **TD vs DP**: TD uses samples (no model). DP uses expected updates (needs model). Both bootstrap.
- **Sarsa vs Q-learning**: Sarsa is on-policy (learns value of exploratory policy = safer). Q-learning is off-policy (learns optimal values regardless of behavior = riskier online).
