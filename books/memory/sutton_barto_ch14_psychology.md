---
name: Sutton & Barto Ch14 - Psychology
description: Chapter 14 summary — classical/instrumental conditioning, Rescorla-Wagner model, TD model of conditioning, blocking, habitual vs goal-directed behavior, model-free vs model-based
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 14: Psychology (pp. 339-370)

## RL Maps to Psychology
- **Prediction** (policy evaluation) <-> **Classical (Pavlovian) conditioning**: Learning stimulus-outcome associations
- **Control** (policy improvement) <-> **Instrumental (operant) conditioning**: Learning action-outcome associations

## Classical Conditioning Terminology
- US (unconditioned stimulus, e.g., food), UR (unconditioned response, e.g., salivation)
- CS (conditioned stimulus, e.g., bell), CR (conditioned response, e.g., salivation to bell)
- ISI: interstimulus interval (CS onset to US onset)

## Blocking (Kamin, p. 344)
Phase 1: light->shock. Phase 2: light+tone->shock. Result: no conditioning to tone. Demonstrates that **surprise (prediction error) is necessary** for learning, not just CS-US pairing.

## Rescorla-Wagner Model (pp. 347-348)
Trial-level error-correction: w_{t+1} = w_t + alpha*delta_t*x(S_t), where delta_t = R_t - v_hat(S_t,w_t)
Explains blocking. Cannot explain within-trial timing effects or second-order conditioning.

## TD Model of Classical Conditioning (pp. 349-353)
Extends Rescorla-Wagner to real-time: delta_t = R_{t+1} + gamma*v_hat(S_{t+1},w) - v_hat(S_t,w)
With eligibility traces: z_{t+1} = gamma*lambda*z_t + x(S_t). Explains ISI effects, second-order conditioning, timing of CR shift to earliest predictor, and why trace conditioning is harder than delay conditioning.

## Instrumental Conditioning (pp. 355-358)
- **Law of Effect** (Thorndike): Satisfying outcomes strengthen responses; discomforting weaken.
- **Shaping** (Skinner): Reinforcing successive approximations to target behavior.
- **Cognitive maps** (Tolman): Internal environment models, distinct from stimulus-response habits.

## Habitual vs Goal-Directed Behavior (pp. 359-365)
- **Habitual**: Stimulus-response, model-free RL. Cached values.
- **Goal-directed**: Deliberate, model-based RL. Uses internal models.
- **Outcome devaluation test**: After training, devalue the reward. Goal-directed animals stop; habitual animals continue. Extensive training shifts behavior from goal-directed to habitual.
