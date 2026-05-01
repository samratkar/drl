---
name: Sutton & Barto Ch9 - On-policy Prediction with Approximation
description: Chapter 9 summary — function approximation, VE objective, SGD, semi-gradient TD, linear methods, feature construction (polynomials, Fourier, coarse coding, tile coding, RBFs), ANNs, LSTD
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
# Chapter 9: On-policy Prediction with Approximation (pp. 195-241)

## Core Problem
Scale RL to large state spaces: v_hat(s, w) ~ v_pi(s), where w in R^d, d << |S|. Key issue: generalization (updating one state affects many others).

## Objective: Mean Squared Value Error (p. 199)
VE(w) = sum_s mu(s)[v_pi(s) - v_hat(s,w)]^2
where mu(s) is the on-policy distribution.

## Stochastic Gradient Descent (p. 201)
w_{t+1} = w_t + alpha[U_t - v_hat(S_t, w_t)] * grad v_hat(S_t, w_t)
- If U_t = G_t (MC return): true gradient, converges to global optimum (linear case)
- If U_t = R_{t+1} + gamma*v_hat(S_{t+1}, w_t) (TD target): **semi-gradient** (ignores gradient through target)

## Semi-gradient TD(0) (p. 203)
w <- w + alpha[R + gamma*v_hat(S', w) - v_hat(S, w)] * grad v_hat(S, w)
Not true gradient descent but converges reliably in the linear case.

## Linear Methods (p. 204)
v_hat(s, w) = w^T x(s). Gradient = x(s). Simple but powerful with good features.

**TD fixed point**: w_TD = A^{-1}b where A = E[x_t(x_t - gamma*x_{t+1})^T], b = E[R_{t+1}*x_t]

**TD error bound** (p. 207): VE(w_TD) <= 1/(1-gamma) * min_w VE(w). Asymptotic TD error is at most 1/(1-gamma) times the best possible.

## Feature Construction Methods (pp. 210-221)

### Polynomials (p. 210)
x_i(s) = product s_j^{c_{i,j}}. Order-n has (n+1)^k features. Not recommended for online RL.

### Fourier Basis (p. 211)
x_i(s) = cos(pi*s^T*c^i). Good general-purpose choice. Outperforms polynomials. Recommended step size: alpha_i = alpha / sqrt(sum c_j^2).

### Coarse Coding (p. 215)
Binary features: state in/out of receptive fields. Width affects initial generalization but not asymptotic accuracy.

### Tile Coding (p. 217) — Most practical for sequential digital computers
Multiple overlapping grid tilings. Exactly n features active per state (one per tiling). alpha = 1/n gives one-trial learning. Asymmetric offsets preferred. Hashing reduces memory.

### RBFs (p. 221)
x_i(s) = exp(-||s-c_i||^2 / (2*sigma_i^2)). Smooth but tile coding often better in high dimensions.

## ANNs (p. 223)
Deep networks learn hierarchical features via backpropagation. Key techniques: dropout, batch normalization, residual learning, convolutions. Universal approximation (one hidden layer) but deep architectures needed in practice.

## LSTD (p. 228)
Most data-efficient linear TD. Directly computes w = A_hat^{-1} b_hat. O(d^2) per step via Sherman-Morrison. No step-size parameter needed. But never forgets (problematic when policy changes).

## Interest and Emphasis (p. 234)
Weight updates by how much we care: w_{t+n} += alpha * M_t * [G_{t:t+n} - v_hat(S_t, w)] * grad v_hat(S_t, w). M_t = I_t + gamma^n * M_{t-n}.
