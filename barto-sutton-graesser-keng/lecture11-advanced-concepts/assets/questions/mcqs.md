---
layout: post
---

# Multiple Choice Questions (Lecture 11: Combined, Advanced PG & Model-Based RL)

1. How does Trust Region Policy Optimization (TRPO) mathematically guarantee monotonic policy improvement?
   a) By clipping the probability ratio to a fixed range.
   b) By adding a KL-divergence constraint on the policy change at each step.
   c) By using a deterministic policy instead of a stochastic one.
   d) By minimizing the MSE of the value network.

2. Why are pure model-based RL methods considered more "sample-efficient" but computationally heavier than model-free methods?
   a) They discard the state representation, requiring less data to optimize.
   b) They can plan and learn using imagined experiences from the model, but require executing search trees or rollouts at decision time.
   c) They only learn value functions.
   d) They use experience replay buffers which are faster to query.

3. During the Simulation (Rollout) phase of standard Monte Carlo Tree Search (MCTS), how are actions selected?
   a) Using the UCT formula.
   b) According to the optimal minimax value.
   c) According to a fast default policy (e.g., random selection) to quickly reach a terminal state.
   d) By querying the value network.

4. In the AlphaGo playout system, how is a leaf node $s_L$ evaluated during MCTS search?
   a) By only running a fast rollout to the end of the game.
   b) By only evaluating the state using the Value Network.
   c) By taking a linear combination (with mixing parameter $\lambda$) of the Value Network output and a fast rollout outcome.
   d) By using the RL policy network to play 100 random games.

5. In MuZero, which function is responsible for predicting the next latent state $s^k$ and the immediate reward $r^k$ from the current latent state $s^{k-1}$ and action $a_k$?
   a) Representation Function ($h$)
   b) Dynamics Function ($g$)
   c) Prediction Function ($f$)
   d) Policy Function ($\pi$)

6. Why is Behavior Cloning highly vulnerable to covariate shift (compounding error)?
   a) It only trains on states visited by the expert, so it doesn't know how to recover when the agent makes a mistake and visits new states.
   b) It uses natural gradients which are unstable.
   c) It requires interactive human labels during test time.
   d) It maximizes policy entropy, leading to high variance.

7. In Generative Adversarial Imitation Learning (GAIL), what represents the surrogate reward $R(s,a)$ maximized by the agent policy (generator)?
   a) $-\ln(1 - D_{\phi}(s, a))$
   b) $\ln D_{\phi}(s, a)$
   c) $-\ln D_{\phi}(s, a)$
   d) $\ln(1 - D_{\phi}(s, a))$


