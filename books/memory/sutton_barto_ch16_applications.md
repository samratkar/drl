---
name: Sutton & Barto Ch16 - Applications and Case Studies
description: Chapter 16 summary — TD-Gammon, Samuel's checkers, Watson Jeopardy, DRAM controller, DQN (Atari), AlphaGo, AlphaGo Zero, personalized web services, thermal soaring
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 16: Applications and Case Studies (pp. 421-457)

## TD-Gammon (Tesauro, pp. 421-426)
- Nonlinear TD(lambda) with ANN. Self-play training. 198 input features encoding backgammon board.
- Architecture: input -> 40-80 hidden units -> sigmoid output (win probability)
- gamma=1, reward only on win/loss. Uses afterstates.
- TD-Gammon 3.0: 80 hidden, expert features, 1.5M games, 3-ply search -> at/above best human level.

## Samuel's Checkers (pp. 426-429)
- Two methods: rote learning (memorize positions) and learning by generalization (linear eval function updated via TD-like method). Precursor to TD-Gammon.

## Watson's Jeopardy Wagering (pp. 429-432)
- Nonlinear TD(lambda) + ANN for state-value function. Millions of simulated games.
- Win rate: 61% baseline -> 67% with learned values.

## DRAM Memory Controller (pp. 432-436)
- Sarsa with tile coding. Implemented on-chip. 7-33% improvement over FR-FCFS.

## DQN — Deep Q-Network (Mnih et al., pp. 436-441)
Three innovations beyond basic Q-learning:
1. **Experience replay**: Store transitions, sample mini-batches
2. **Target network**: Frozen copy updated every C steps. Stabilizes targets.
3. **Reward clipping**: Clip to [-1, 1]

Architecture: 3 conv layers (32, 64, 64 feature maps) -> 512 FC -> 18 action outputs.
Input: 84x84x4 stacked grayscale frames. Human-level on 29/46 Atari games. Poor on deep-planning games (Montezuma's Revenge).

## AlphaGo (Silver et al., pp. 441-447)
Pipeline: (1) SL policy network (13-layer CNN, 57% accuracy), (2) RL policy network (policy gradient self-play), (3) Value network (MC evaluation of RL policy), (4) Fast rollout policy.
APV-MCTS: expansion guided by SL policy, evaluation mixes value net + rollout (best at eta=0.5).
Defeated Fan Hui 5-0, Lee Sedol 4-1.

## AlphaGo Zero (Silver et al., pp. 447-450)
- **No human data**. Only rules of Go.
- Single two-headed CNN (41 conv layers + batch norm + skip connections -> policy head + value head)
- Input: 19x19x17 binary feature planes. Simpler MCTS (no rollouts).
- MCTS as "policy improvement operator." 4.9M self-play games, ~3 days.
- Elo 5185 vs AlphaGo Lee-Sedol version 3739. Defeated AlphaGo 100-0.

## Personalized Web (pp. 450-453)
CTR (clicks/visits) vs LTV (clicks/visitors). Fitted Q Iteration (batch RL) for LTV optimization outperformed greedy CTR optimization on long-term engagement.

## Thermal Soaring (pp. 453-457)
One-step Sarsa + state aggregation. 4D state, 9 actions. Vertical wind acceleration and torque sufficient for effective soaring. Learned spiral flight in thermals.
