# Programming Questions (Lecture 8: Deep Q-Learning)

**Question 1: Implementing the DDQN Target in PyTorch**
Assume you have a batch of transitions. You are given the following PyTorch tensors:
* `rewards`: A tensor of shape `(batch_size, 1)`
* `dones`: A boolean tensor of shape `(batch_size, 1)` indicating terminal states.
* `gamma`: A float scalar.
* `online_q_next`: A tensor of shape `(batch_size, num_actions)` containing the Q-values for the next states from the Online Network.
* `target_q_next`: A tensor of shape `(batch_size, num_actions)` containing the Q-values for the next states from the Target Network.

Write the PyTorch code (2-3 lines) to compute the DDQN target tensor `y_j`.
