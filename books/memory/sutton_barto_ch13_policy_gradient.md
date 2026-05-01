---
name: Sutton & Barto Ch13 - Policy Gradient Methods
description: Chapter 13 summary — policy gradient theorem, REINFORCE, baselines, actor-critic, eligibility traces for policy gradients, Gaussian policies for continuous actions
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 13: Policy Gradient Methods (pp. 321-338)

## Core Idea
Learn a parameterized policy pi(a|s,theta) directly, without consulting a value function for action selection. Update theta by gradient ascent on performance J(theta).

## Policy Gradient Theorem (p. 324-326) — CENTRAL RESULT
grad J(theta) proportional_to sum_s mu(s) sum_a q_pi(s,a) * grad pi(a|s,theta)

Remarkable: gradient of performance does NOT involve derivative of the state distribution mu(s), even though mu depends on theta. Makes gradient estimation from experience alone possible.
- Episodic: proportionality constant = average episode length
- Continuing: it's an equality

## Policy Parameterizations

### Discrete Actions — Softmax in Preferences (p. 322)
pi(a|s,theta) = exp(h(s,a,theta)) / sum_b exp(h(s,b,theta))
Often h(s,a,theta) = theta^T x(s,a) (linear preferences)
Eligibility vector: grad ln pi(a|s,theta) = x(s,a) - sum_b pi(b|s,theta)*x(s,b)

### Continuous Actions — Gaussian Policy (p. 335)
pi(a|s,theta) = Normal(mu(s,theta), sigma(s,theta)^2)
mu = theta_mu^T x_mu(s), sigma = exp(theta_sigma^T x_sigma(s))

## Algorithms

### REINFORCE (p. 328)
theta <- theta + alpha * gamma^t * G_t * grad ln pi(A_t|S_t, theta)
Monte Carlo policy gradient. Unbiased estimate of gradient direction. High variance.

### REINFORCE with Baseline (p. 330)
theta <- theta + alpha * gamma^t * (G_t - b(S_t)) * grad ln pi(A_t|S_t, theta)
b(s) is any function not depending on a. Does not bias the gradient (because sum_a b(s)*grad pi = 0). Natural choice: b(s) = v_hat(S_t, w). Dramatically reduces variance.

### One-step Actor-Critic (p. 332)
theta <- theta + alpha * I * delta_t * grad ln pi(A_t|S_t, theta)
w <- w + alpha^w * delta_t * grad v_hat(S_t, w)
delta_t = R + gamma*v_hat(S',w) - v_hat(S,w). Uses bootstrapping (critic) unlike REINFORCE.

### Actor-Critic with Eligibility Traces (p. 332)
z^theta <- gamma*lambda^theta*z^theta + I*grad ln pi(A|S,theta)
z^w <- gamma*lambda^w*z^w + grad v_hat(S,w)
theta <- theta + alpha^theta*delta*z^theta
w <- w + alpha^w*delta*z^w

### Continuing Actor-Critic (p. 333)
Uses average reward R_bar and differential TD error: delta = R - R_bar + v_hat(S',w) - v_hat(S,w)

## Key Distinctions
- **REINFORCE with baseline vs actor-critic**: Baseline is NOT a critic. Critic bootstraps (creates bias, reduces variance). Baseline only reduces variance without bias.
- **Policy gradient vs action-value methods**: Policy gradient can learn stochastic optimal policies, handle continuous actions naturally, and has stronger convergence guarantees (smooth policy changes).
- **Short corridor example** (p. 323): Function approximation forces epsilon-greedy to choose between two bad deterministic policies. Policy gradient finds the optimal stochastic policy (p(right) ~ 0.59).
