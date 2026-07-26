---
layout: post
---

# Subjective Solutions (Lecture 11: Combined, Advanced PG & Model-Based RL)

1. **Model-Free vs. Model-Based RL:**
   * **Difference:** Model-free RL learns directly from empirical outcomes of actions taken in the environment (e.g. updating Q-values or policy gradients). Model-based RL attempts to learn or use a model of the transition dynamics $P(s'|s,a)$ and rewards $R(s,a)$ to simulate future trajectories offline.
   * **Sample Efficiency:** Model-based RL is more sample-efficient because it can perform "imagined" updates inside the simulated model, reducing the number of costly real-world environment interactions.
   * **Risk/Challenge:** The primary risk is **model error compounding (trajectory drift)**. Inaccuracies in the transition dynamics accumulate as predictions are made further into the future, leading to highly inaccurate simulated planning.

2. **The MCTS Framework:**
   * **Selection**: Traverse the tree from the root using a selection policy (like UCT) until a leaf node with unexpanded actions is reached.
   * **Expansion**: Expand the leaf node by adding one of its unexplored legal actions as a child node.
   * **Simulation**: Run a fast rollout simulation from the new child node using a default rollout policy until a terminal state is reached.
   * **Backpropagation**: Propagate the terminal reward back up to the root, incrementing node visit counts and updating the running average value of the nodes.
   * **Value Evaluation**: MCTS evaluates the value of a state dynamically by averaging the actual empirical outcomes of playouts that passed through that state, rather than using a static heuristic or lookup table.

3. **AlphaGo Network Design:**
   * **SL Policy Network**: Pre-trained on human games to predict expert moves. Its probabilities serve as prior probabilities $P(s,a)$ to guide selection in MCTS.
   * **RL Policy Network**: Trained via self-play policy gradients. Used to generate games to train the Value Network.
   * **Value Network**: Predicts winning probability from a state, providing a value estimate $v_{\theta}(s_L)$ for MCTS leaf evaluation.
   * **Rollout Policy**: A very fast, lightweight policy used to simulate playouts to terminal states in MCTS.
   * **Integration**: In MCTS selection, prior probabilities come from $\pi_{SL}$. In leaf node evaluation, the estimate from $v_{\theta}$ is mixed 50/50 with the fast simulation outcome from $\pi_{\text{rollout}}$.

4. **AlphaZero vs. MuZero Planning:**
   * **AlphaZero** plans using a ground-truth simulator that embodies the exact rules of the environment. Its search tree contains real game states, and its neural network evaluates leaf nodes directly.
   * **MuZero** plans inside a learned model without knowing the environment rules. Its search tree is built in a latent (hidden) space. 
   * **Role of Networks in MuZero MCTS:**
     * **Representation ($h_{\theta}$)**: Encodes historical real observations to get the root latent state $s^0$.
     * **Dynamics ($g_{\theta}$)**: Performs virtual expansions inside MCTS, transitioning from $s^{k-1}$ to $s^k$ and predicting immediate reward $r^k$ without querying the environment.
     * **Prediction ($f_{\theta}$)**: Evaluates latent states $s^k$ to output policy priors $\mathbf{p}^k$ and value estimates $v^k$ for UCT selection.

5. **Behavior Cloning vs. DAgger:**
   * **Covariate Shift / Compounding Error**: Behavior Cloning is trained offline on states visited by the expert. At execution time, any small error by the agent leads it to a state that is outside its training distribution (covariate shift). The agent makes worse decisions in these unfamiliar states, leading to compounding errors that drift it completely off course.
   * **How DAgger Fixes This**: DAgger is an interactive algorithm. It runs the agent's current policy in the environment to collect the states the agent actually visits (including those reached after mistakes). It then queries the expert to label those states with correct actions. By aggregating these new state-action pairs into the training dataset and retraining, the agent learns how to recover from its own errors.

6. **Inverse Reinforcement Learning and the Ambiguity Problem:**
   * **Goal of IRL**: IRL aims to recover the underlying reward function $R^*(s,a)$ optimized by the expert from a set of expert demonstrations $\mathcal{D}$. Once learned, this reward function can be used to train an agent via standard reinforcement learning.
   * **Reward Ambiguity Problem**: An expert policy can be optimal under infinitely many reward functions. For instance, a trivial reward function $R(s,a) = 0$ for all $(s,a)$ makes every policy (including the expert's) optimal. 
   * **Maximum Entropy IRL Approach**: To resolve this ambiguity, Maximum Entropy IRL uses information theory. It selects the reward function that makes the expert policy optimal while maximizing the entropy of the trajectory distribution (which means it makes the fewest additional assumptions about trajectory preferences other than matching the expert's feature expectations).

7. **GAIL and the Minimax Game:**
   * **GAN Analogy**: GAIL frames imitation learning as a two-player minimax game:
     * **Generator (Agent Policy $\pi_{\theta}$)**: Attempts to generate state-action transitions that look indistinguishable from the expert demonstrations.
     * **Discriminator ($D_{\phi}(s, a)$)**: Attempts to classify whether a given transition $(s,a)$ came from the expert ($D \rightarrow 1$) or the agent ($D \rightarrow 0$).
   * **Surrogate Reward Connection**: The generator is updated via policy gradient methods (e.g. TRPO/PPO) to maximize a surrogate reward $R(s,a) = -\ln(1 - D_{\phi}(s,a))$, which encourages the agent to visit states and take actions that fool the discriminator into thinking they are expert-generated.


