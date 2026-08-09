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

5. **Behavior Cloning Gradient Update & Covariate Shift Analysis:**
   Consider a 1D continuous lane-centering task with state $x \in \mathbb{R}$ (lateral displacement) and discrete steering actions $a \in \{0, 1\}$ ($0$: Steer Left, $1$: Steer Right).
   The agent policy is parameterized as $P_\theta(a=1 \mid x) = \sigma(w \cdot x) = \frac{1}{1 + e^{-w \cdot x}}$.
   The expert target action for state $x > 0$ is $a^* = 0$ (steer left).
   Given an expert demonstration sample $(x = 0.4, a^* = 0)$ and initial weight $w_0 = -0.5$:
   * Compute the policy probability $P(a=0 \mid 0.4)$ and the Cross-Entropy loss $\mathcal{L}_{CE} = -\ln P(a=0 \mid 0.4)$.
   * Calculate the loss gradient $\frac{\partial \mathcal{L}}{\partial w} = (P(a=1 \mid x) - \mathbb{I}(a^*=1)) \cdot x$.
   * Perform one step of gradient descent with learning rate $\alpha = 2.0$ to find updated weight $w_1$.
   * If an unobserved disturbance pushes the vehicle to an out-of-distribution state $x = +4.0$, compute the loss gradient magnitude at $x = +4.0$ compared to $x = +0.4$ for initial weight $w = 0$, and explain how DAgger leverages this signal to eliminate compounding error.



