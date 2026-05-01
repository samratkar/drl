---
name: Sutton & Barto Ch8 - Planning and Learning
description: Chapter 8 summary — models (distribution vs sample), Dyna-Q, Dyna-Q+, prioritized sweeping, expected vs sample updates, unified view of planning and learning
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 8: Planning and Learning with Tabular Methods (pp. 159-194)

## Core Insight
Planning (using a model) and learning (from experience) share the same computational mechanism: value-function updates via backups. They differ only in the source of experience (simulated vs real).

## Models (p. 159-160)
- **Distribution model**: Produces all possible transitions with probabilities (e.g., p(s',r|s,a) in DP)
- **Sample model**: Produces one sampled transition. Easier to obtain; sufficient for most planning.

## Dyna Architecture

### Dyna-Q (p. 164)
Integrates three processes per real step:
1. **Direct RL**: Q(S,A) += alpha[R + gamma*max_a Q(S',a) - Q(S,A)] on real experience
2. **Model learning**: Model(S,A) <- R,S' (deterministic assumption)
3. **Planning**: n times: pick random previously-seen (S,A), get R,S' from model, do Q-learning update

Dyna maze example: n=0 (pure Q-learning) ~25 episodes; n=5 ~5 episodes; n=50 ~3 episodes.

### Dyna-Q+ (p. 167-168)
Adds exploration bonus: reward = r + kappa*sqrt(tau), where tau = time steps since (s,a) last tried in real interaction. Encourages revisiting long-untried actions. Essential for discovering environmental improvements (shortcut maze example: regular Dyna-Q never finds the shortcut).

## Prioritized Sweeping (p. 168-170)
Focus updates on states whose values would change most. Priority = |TD error|. Propagate backward from changed states. 5-10x more efficient than uniform random selection on maze tasks.

## Expected vs Sample Updates (p. 172)
Three binary dimensions classify one-step updates:
1. State values vs action values
2. Optimal (v_*, q_*) vs arbitrary policy (v_pi, q_pi)  
3. Expected vs sample updates

Eight cases, seven map to named algorithms:
- q_* expected: q-value iteration | q_* sample: **Q-learning**
- q_pi expected: q-policy evaluation | q_pi sample: **Sarsa**
- v_* expected: value iteration | v_pi expected: DP policy evaluation | v_pi sample: **TD(0)**

## Key Distinctions
- **Direct vs indirect RL**: Direct = experience -> values. Indirect = experience -> model -> values (more sample-efficient but depends on model quality). Dyna combines both.
- **Model-based vs model-free**: Model-based relies on planning; model-free relies on direct learning from experience.
- **Backward focusing** (prioritized sweeping) vs **forward focusing** (focus on states reachable from frequently visited states under current policy)
