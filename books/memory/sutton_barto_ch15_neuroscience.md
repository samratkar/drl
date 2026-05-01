---
name: Sutton & Barto Ch15 - Neuroscience
description: Chapter 15 summary — reward prediction error hypothesis, dopamine as TD error, basal ganglia actor-critic, Schultz experiments, STDP, hedonistic neurons, collective RL
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 15: Neuroscience (pp. 377-419)

## Reward Prediction Error (RPE) Hypothesis (pp. 374-383)
Phasic activity of dopamine neurons (SNpc/VTA) corresponds to TD errors:
1. Positive phasic response to **unexpected** rewards
2. No response to **fully predicted** rewards
3. Negative response (below baseline) when expected reward **omitted**
4. Response **shifts** from reward to earliest predictive stimulus as learning progresses

Dopamine is a **reinforcement signal** (delta_t), NOT a reward signal (R_t).

## Schultz's Experiments (pp. 374-383)
Monkey dopamine neuron recordings showing all four TD-error correspondences. Optogenetic experiments confirm causal role.

## Basal Ganglia as Actor-Critic (pp. 383-400)
- **Striatum**: Main input structure. Medium spiny neurons (MSNs).
- **Ventral striatum**: Critic — learns value function v_hat
- **Dorsal striatum**: Actor — learns policy. DLS = habitual/model-free; DMS = goal-directed/model-based
- **Dopamine**: TD error signal broadcast to both via extensive axonal arbors

## Learning Rules
- **Critic** (two-factor): Interaction between delta and presynaptic traces (non-contingent eligibility)
- **Actor** (three-factor): Interaction between delta, presynaptic, and postsynaptic activity (contingent eligibility)

## Spike-Timing-Dependent Plasticity (STDP) (pp. 401-402)
Synaptic changes depend on pre/post spike timing. **Reward-modulated STDP**: changes only persist when dopamine arrives within a time window (up to 10s). Neural mechanism for three-factor actor learning.

## Hedonistic Neurons (Klopf, pp. 402-404)
Individual neurons as RL agents maximizing rewarding input. Introduced concept of synaptic eligibility.

## Collective RL (pp. 404-407)
Teams sharing common reward signal. With contingent eligibility traces, teams of Bernoulli-logistic REINFORCE units implement policy gradient. Solves structural credit assignment problem.

## Additional Neural Substrates
- **Hippocampus**: Forward sweeps at choice points — supports model-based planning
- **Orbitofrontal cortex (OFC)**: Encodes reward component of environment models
- **Addiction** (p. 409): Drugs produce dopamine surges that can't be predicted away by TD learning, causing unbounded value growth for drug-associated states
