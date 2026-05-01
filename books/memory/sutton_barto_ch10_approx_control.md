---
name: Sutton & Barto Ch10 - On-policy Control with Approximation
description: Chapter 10 summary — semi-gradient Sarsa, episodic/continuing control, average reward setting, differential returns, futility of discounting with function approximation
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 10: On-policy Control with Approximation (pp. 243-256)

## Semi-gradient Sarsa (p. 244)
q_hat(s,a,w) ~ q_pi(s,a). Update:
w <- w + alpha[R + gamma*q_hat(S',A',w) - q_hat(S,A,w)] * grad q_hat(S,A,w)
Combined with epsilon-greedy policy improvement.

## n-step Semi-gradient Sarsa (p. 247)
Uses n-step return: G_{t:t+n} = R_{t+1} + ... + gamma^{n-1}*R_{t+n} + gamma^n*q_hat(S_{t+n},A_{t+n},w)
Intermediate n (4 or 8) best on Mountain Car task.

## Average Reward Setting (p. 249)
For continuing tasks. Quality = average reward rate r(pi) = lim (1/h) sum E[R_t].
No discounting — immediate and delayed rewards equally important. Requires ergodicity.

**Differential return**: G_t = (R_{t+1} - r(pi)) + (R_{t+2} - r(pi)) + ...
**Differential TD error**: delta_t = R_{t+1} - R_bar + q_hat(S',A',w) - q_hat(S,A,w)

### Differential Semi-gradient Sarsa (p. 251)
Two step-sizes: alpha (weights), beta (average reward estimate R_bar).
- delta <- R - R_bar + q_hat(S',A',w) - q_hat(S,A,w)
- R_bar <- R_bar + beta*delta
- w <- w + alpha*delta*grad q_hat(S,A,w)

## Futility of Discounting (p. 253-254)
**Key result**: J(pi) = sum_s mu_pi(s)*v_pi^gamma(s) = r(pi)/(1-gamma). Discount rate gamma has NO effect on policy ordering under function approximation. Discounting becomes merely a solution-method parameter, not a problem parameter.

**Loss of policy improvement theorem** with approximation: improving discounted value of one state no longer guarantees overall policy improvement.

## Key Example
**Mountain Car** (p. 244): Underpowered car builds momentum oscillating between slopes. 3 actions, reward -1/step. Tile coding with 8 tilings. alpha=0.5/8 best. Optimistic initialization (all zeros when true values negative) drives exploration.
