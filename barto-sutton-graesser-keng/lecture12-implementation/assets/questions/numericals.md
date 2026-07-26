---
layout: post
---

# Numerical Questions (Lecture 12: Implementation Frameworks)

1. **Trial Step Count Calculation:**
   A researcher configures an SLM Lab experiment:
   * The training runs for 500 episodes.
   * On average, each episode lasts 200 environment steps before termination or truncation.
   * To account for statistical variance, the researcher runs a **Trial** containing 5 independent Sessions with different random seeds.
   Calculate the total number of environment step interactions (`env.step`) executed across the entire Trial.

2. **DQN Target Value Calculation:**
   During a DQN update step in SLM Lab:
   * The reward received is $R = -1.0$.
   * The discount factor is $\gamma = 0.99$.
   * The next state $S'$ is terminal (meaning `terminated = True`, `truncated = False`).
   * The online network predicts $Q(S', a) = [0.5, 0.9, -0.2]$.
   * The target network predicts $Q^-(S', a) = [0.4, 0.8, -0.3]$.
   Calculate the target Q-value $y$ that will be used to compute the loss for the transition $(S, A, R, S')$.
