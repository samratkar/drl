---
name: Sutton & Barto Ch1 - Introduction
description: Chapter 1 summary — RL definition, four elements (policy/reward/value/model), exploration vs exploitation, tic-tac-toe example, three historical threads
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 1: Introduction (pp. 1-22)

## Core Definition
RL = learning what to do (map situations to actions) to maximize a numerical reward signal. Two distinguishing characteristics: **trial-and-error search** and **delayed reward**. RL is a third paradigm alongside supervised and unsupervised learning.

## Exploration vs Exploitation (p. 3)
Central dilemma unique to RL. Must exploit known good actions while exploring potentially better ones. Neither alone suffices.

## Four Elements of an RL System (pp. 6-7)
1. **Policy**: Mapping from states to actions (or distributions). Core of the agent.
2. **Reward signal**: Scalar from environment each step. Defines the goal. Immediate goodness.
3. **Value function**: Expected cumulative future reward. Long-term desirability. Actions are chosen based on values. "The central role of value estimation is arguably the most important thing learned about RL over the last six decades."
4. **Model** (optional): Predicts environment response. Enables planning. Model-based vs model-free distinction.

## Tic-Tac-Toe Example (pp. 8-12)
- Table of state values (probability of winning). V(s) <- V(s) + alpha * [V(S_{t+1}) - V(S_t)]
- Illustrates TD learning: updates based on difference between successive estimates
- Unlike minimax (no opponent model needed), evolutionary methods (learns during play, evaluates states not policies), DP (no complete model needed)

## Three Historical Threads (pp. 13-21)
1. **Trial-and-error learning**: Thorndike's Law of Effect (1911) -> Klopf (1972) -> Barto/Sutton
2. **Optimal control / DP**: Bellman (1957), Howard's policy iteration (1960), curse of dimensionality
3. **Temporal-difference learning**: Samuel's checkers (1959), Sutton TD(lambda) (1988), Watkins Q-learning (1989) unified all three threads

Actor-critic architecture emerged ~1981 combining TD with trial-and-error.
