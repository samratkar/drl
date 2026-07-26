---
layout: post
---

# Subjective Solutions (Lecture 12: Implementation Frameworks)

1. **Gymnasium vs. SLM Lab Roles:**
   * **Gymnasium** defines the environment's state/action spaces and transition logic. It accepts an action, executes a step inside the simulator, and outputs the next observation, reward, and done flags. It has no memory of past states and contains no neural networks.
   * **SLM Lab** sits above Gymnasium. It defines the agent's policy, wraps the neural networks in PyTorch, manages training batches in memory, and implements the algorithm's loss function (e.g. policy gradient loss). The policy gradient logic is executed inside the SLM Lab `Algorithm` component (e.g. `reinforce.py` or `ppo.py`), which calls Gymnasium's `step` function to gather transition data.

2. **The Session-Trial-Experiment Hierarchy:**
   * **Session:** A single execution of a configuration with a single seed.
   * **Trial:** Group of sessions running the exact same configuration but with different seeds.
   * **Experiment:** Container for multiple trials, usually sweeping over hyperparameters.
   * **Importance of Trial:** DRL algorithms are highly sensitive to initial seed conditions (e.g. weight initialization, environment randomness). Running a single session can lead to false conclusions. Averaging across multiple sessions in a Trial provides a statistically robust picture of performance.

3. **DQN vs. Double DQN Implementation in SLM Lab:**
   * In standard **DQN**, the target value is computed using the target network's maximum predicted Q-value: $Y = R + \gamma \max_{a'} Q^-(S', a')$.
   * In **Double DQN**, selection and evaluation are decoupled. The online network selects the best action for the next state, and the target network evaluates that action: $Y = R + \gamma Q^-\left(S', \text{argmax}_{a'} Q(S', a')\right)$.
   * In SLM Lab, both are handled in `slm_lab/agent/algorithm/dqn.py`. Double DQN is activated by setting `"val_spec": {"val_type": "DoubleDQN"}` or similar config in the agent spec JSON file, which triggers the decoupled target calculation in `compute_q_targets`.
