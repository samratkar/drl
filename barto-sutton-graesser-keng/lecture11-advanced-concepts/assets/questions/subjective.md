---
layout: post
---

# Subjective Questions (Lecture 11: Combined, Advanced PG & Model-Based RL)

1. **Model-Free vs. Model-Based RL:**
   Describe the difference between model-free and model-based reinforcement learning. Why is model-based RL generally more sample-efficient, and what is the primary risk/challenge of planning with a learned model?

2. **The MCTS Framework:**
   Explain the four phases of the Monte Carlo Tree Search (MCTS) framework (Selection, Expansion, Simulation, Backpropagation). How does MCTS evaluate the value of a state without a global value function?

3. **AlphaGo Network Design:**
   Explain the roles of the SL Policy Network, the RL Policy Network, the Value Network, and the Rollout Policy in AlphaGo. How do these components work together inside the MCTS framework during a game?

4. **AlphaZero vs. MuZero Planning:**
   Describe how MuZero plans inside a learned model compared to how AlphaZero plans. Detail the roles of the Representation, Dynamics, and Prediction networks in MuZero's tree search.

5. **Behavior Cloning vs. DAgger:**
   Explain the concept of covariate shift in Behavior Cloning and how it leads to compounding errors. How does DAgger (Dataset Aggregation) address this issue?

6. **Inverse Reinforcement Learning and the Ambiguity Problem:**
   Explain the goal of Inverse Reinforcement Learning (IRL). What is the reward ambiguity problem in IRL, and how does Maximum Entropy IRL address this?

7. **GAIL and the Minimax Game:**
   Describe how Generative Adversarial Imitation Learning (GAIL) adapts the concept of GANs to imitation learning. What are the roles of the policy and the discriminator in this framework?


