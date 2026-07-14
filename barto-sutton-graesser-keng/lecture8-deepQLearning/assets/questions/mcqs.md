# Multiple Choice Questions (Lecture 8: Deep Q-Learning)

1. What is the primary purpose of Experience Replay in DQN?
   a) To increase the learning rate of the network.
   b) To allow the agent to look further into the future before bootstrapping.
   c) To break the temporal correlations in sequential data, making it pseudo-i.i.d.
   d) To prevent the target from moving during training.

2. In standard DQN, how is the "moving target" problem mitigated?
   a) By using a separate Target Network that is only updated periodically.
   b) By reducing the discount factor $\gamma$.
   c) By using a highly specialized Convolutional Neural Network.
   d) By decoupling action selection and action evaluation.

3. Why does the overestimation bias occur in standard Q-learning?
   a) Because neural networks naturally initialize with very high weights.
   b) Due to the $\max$ operator applied to noisy Q-value estimates.
   c) Because Experience Replay samples the most rewarding transitions too often.
   d) Because the Target Network lags behind the Online Network.

4. How does Double DQN (DDQN) solve the overestimation bias?
   a) By removing the Target Network entirely.
   b) By taking the minimum over two separate neural networks instead of the maximum.
   c) By using the Online Network to select the action and the Target Network to evaluate it.
   d) By using the Target Network to select the action and the Online Network to evaluate it.

5. In DDQN, if the Online Network overestimates the value of a sub-optimal action and selects it, what prevents the bias from propagating?
   a) The Replay Buffer simply discards that transition.
   b) The Target Network (which evaluates the action) is unlikely to have the exact same overestimation error for that action.
   c) The learning rate is dynamically lowered.
   d) The $\max$ operator filters out the selection error.
