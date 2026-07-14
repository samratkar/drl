# Multiple Choice Questions (Lecture 9: Policy Gradients)

1. Which of the following is a primary advantage of Policy Gradient methods over Action-Value methods like DQN?
   a) Policy Gradients always converge to a deterministic policy.
   b) Policy Gradients can naturally handle continuous action spaces.
   c) Policy Gradients do not require a neural network.
   d) Policy Gradients have lower variance than Q-learning.

2. In the REINFORCE algorithm, what is the purpose of the $\nabla_{\theta} \ln \pi(A_t|S_t, \theta)$ term?
   a) It computes the exact value of the state.
   b) It determines the direction in weight space that increases the probability of action $A_t$.
   c) It forces the policy to become deterministic.
   d) It calculates the TD error.

3. Why is a baseline $b(s)$ subtracted from the return $G_t$ in Policy Gradients?
   a) To reduce the variance of the gradient estimates without introducing bias.
   b) To increase the learning rate over time.
   c) To make the algorithm off-policy.
   d) To ensure all returns are strictly positive.

4. In an Actor-Critic architecture, what is the role of the Critic?
   a) To select the final actions the agent takes in the environment.
   b) To estimate the Value function $V(s)$ and compute the TD Error (Advantage) for the Actor.
   c) To store transitions in the Replay Buffer.
   d) To compute the log probabilities of actions.
