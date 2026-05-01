---
name: Sutton & Barto Book Overview
description: Structure and chapter map of "Reinforcement Learning: An Introduction" 2nd edition (Sutton & Barto, 2018) — use as entry point when user asks about the book
type: reference
originSessionId: 8c4d2f9c-976e-4ade-88ca-2bef48ec669d
---
**Book**: Reinforcement Learning: An Introduction, 2nd Edition
**Authors**: Richard S. Sutton and Andrew G. Barto (2018, MIT Press)
**Location**: /Users/samrat.kar/git/drl/books/BartoSutton.pdf

## Three-Part Structure

**Part I — Tabular Solution Methods** (Chapters 2-8): Complete RL in the tabular case (exact solutions possible).
- Ch 2: Multi-armed Bandits (k-armed bandits, epsilon-greedy, UCB, gradient bandits)
- Ch 3: Finite MDPs (agent-environment interface, Bellman equations, optimal value functions)
- Ch 4: Dynamic Programming (policy evaluation, policy iteration, value iteration, GPI)
- Ch 5: Monte Carlo Methods (first-visit/every-visit MC, on/off-policy, importance sampling)
- Ch 6: Temporal-Difference Learning (TD(0), Sarsa, Q-learning, Expected Sarsa, Double Q-learning)
- Ch 7: n-step Bootstrapping (n-step TD, n-step Sarsa, tree backup, Q(sigma))
- Ch 8: Planning and Learning (Dyna-Q, prioritized sweeping, MCTS, expected vs sample updates)

**Part II — Approximate Solution Methods** (Chapters 9-13): Function approximation for large state spaces.
- Ch 9: On-policy Prediction with Approximation (SGD, semi-gradient TD, linear methods, tile coding, ANNs, LSTD)
- Ch 10: On-policy Control with Approximation (semi-gradient Sarsa, average reward, differential returns)
- Ch 11: Off-policy Methods with Approximation (deadly triad, Baird's counterexample, divergence)
- Ch 12: Eligibility Traces (lambda-return, TD(lambda), true online TD(lambda), Sarsa(lambda))
- Ch 13: Policy Gradient Methods (REINFORCE, actor-critic, policy gradient theorem, Gaussian policies)

**Part III — Looking Deeper** (Chapters 14-17): Connections and frontiers.
- Ch 14: Psychology (classical/instrumental conditioning, Rescorla-Wagner, TD model, habitual vs goal-directed)
- Ch 15: Neuroscience (dopamine as TD error, basal ganglia actor-critic, reward prediction error hypothesis)
- Ch 16: Applications (TD-Gammon, DQN, AlphaGo/Zero, Watson Jeopardy, thermal soaring)
- Ch 17: Frontiers (GVFs, options/temporal abstraction, POMDPs, intrinsic motivation)

## Key Notation (pp. xix-xx)
- S_t, A_t, R_t: state, action, reward at time t (random variables, capitals)
- s, a, r: their instantiations (lowercase)
- pi(a|s): policy (stochastic); pi(s): policy (deterministic)
- v_pi(s): state-value function under pi
- q_pi(s,a): action-value function under pi
- v_*(s), q_*(s,a): optimal value functions
- G_t: return from time t
- gamma: discount rate; alpha: step size; epsilon: exploration parameter; lambda: trace decay
- p(s',r|s,a): dynamics function (four-argument)
- rho: importance sampling ratio
- w: weight vector for function approximation; x(s): feature vector
- delta_t: TD error
