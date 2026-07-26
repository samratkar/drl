---
layout: post
---

# Numerical Questions (Lecture 11: Combined, Advanced PG & Model-Based RL)

1. **UCT Value Calculation:**
   Given a node $s$ with total visit count $N(s) = 100$. A child action node $a$ has a visit count of $N(s,a) = 20$ and an accumulated action value of $W(s,a) = 12.0$.
   Using the UCT exploration parameter $c = 1.414$:
   * Calculate the average action value $Q(s,a)$.
   * Calculate the exploration bonus.
   * Calculate the total UCT score for action $a$.

2. **AlphaGo Leaf Evaluation:**
   During a match, AlphaGo reaches a leaf node $s_L$.
   * The Value Network predicts a win probability (expected value) of $v_{\theta}(s_L) = 0.75$.
   * The fast rollout simulation leads to a loss ($z = -1.0$).
   * The mixing parameter is set to $\lambda = 0.5$.
   Calculate the final combined leaf evaluation $V(s_L)$ that will be backpropagated up the tree.

3. **AlphaZero Loss Calculation:**
   Given a self-play game outcome $z = +1.0$ (win). At step $t$, the MCTS search output is $\boldsymbol{\pi}_t = [0.1, 0.7, 0.2]^T$ for three discrete actions. The neural network's policy head outputs $\mathbf{p}_t = [0.2, 0.5, 0.3]^T$ and the value head predicts $v_t = 0.6$.
   * Compute the squared error loss for the value head.
   * Compute the cross-entropy loss for the policy head ($-\sum_a \pi_a \ln p_a$).
   * Calculate the total loss at this step (excluding L2 regularization).

4. **GAIL Surrogate Reward Calculation:**
   An agent state-action pair $(s,a)$ is evaluated by the discriminator $D_{\phi}(s, a)$.
   * Calculate the surrogate reward $R(s,a) = -\ln(1 - D_{\phi}(s,a))$ when the discriminator is highly confident the transition is expert-like: $D_{\phi}(s,a) = 0.90$.
   * Calculate the surrogate reward $R(s,a)$ when the discriminator is confident the transition is agent-like: $D_{\phi}(s,a) = 0.10$.
   *(Note: $\ln 0.1 \approx -2.302$, $\ln 0.9 \approx -0.105$)*


