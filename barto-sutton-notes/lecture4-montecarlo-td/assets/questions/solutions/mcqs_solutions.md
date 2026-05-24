---
layout: post
---

# MCQ Solutions - Lecture 4

1. Monte Carlo (MC) methods are characterized by:
   A. Learning from individual steps through bootstrapping.
   B. Requiring a complete model of environment dynamics.
   C. Learning from complete episodes of experience.
   D. Being applicable only to non-episodic tasks.

   **Answer: C.** Learning from complete episodes. MC methods wait for the episode to end before updating estimates.

2. In First-visit MC prediction, the return for state $s$ is averaged based on:
   A. Only the very first visit to $s$ in the entire training process.
   B. The first time $s$ is visited in each episode.
   C. Every time $s$ is visited in an episode.
   D. The last visit to $s$ before termination.

   **Answer: B.** The first time $s$ is visited in each episode. Every-visit MC (C) would average all visits.

3. The "Exploring Starts" assumption in MC Control is used to:
   A. Speed up the bootstrapping process.
   B. Ensure that every state-action pair is visited infinitely often.
   C. Eliminate the need for action-value estimates $Q(s, a)$.
   D. Reduce the variance of Importance Sampling.

   **Answer: B.** Ensure every state-action pair is visited. This solves the exploration problem without using stochastic policies.

4. On-policy methods evaluate or improve:
   A. A greedy policy while following a random policy.
   B. The same policy that is used to make decisions.
   C. A deterministic policy from stochastic data.
   D. Multiple target policies simultaneously.

   **Answer: B.** The same policy used for decisions. On-policy learning is "self-evaluating."

5. Off-policy learning via Importance Sampling requires "Coverage," which means:
   A. $\pi(a\mid s) > 0 \implies b(a\mid s) > 0$.
   B. $b(a\mid s) > 0 \implies \pi(a\mid s) > 0$.
   C. The behavior policy must be deterministic.
   D. The target policy must be $\epsilon$-greedy.

   **Answer: A.** Target actions must be possible under behavior policy. You cannot learn about an action if you never see it.

6. Which type of Importance Sampling is generally preferred due to its lower variance?
   A. Ordinary Importance Sampling
   B. Weighted Importance Sampling
   C. Direct Importance Sampling
   D. Sequential Importance Sampling

   **Answer: B.** Weighted Importance Sampling. It has lower variance than ordinary IS, although it is slightly biased (toward 0).

7. Temporal-Difference (TD) learning is a combination of:
   A. MC and Dynamic Programming.
   B. Supervised and Unsupervised learning.
   C. Bandits and Monte Carlo.
   D. Planning and Search.

   **Answer: A.** MC and Dynamic Programming. TD learns from experience (like MC) and bootstraps (like DP).

8. The TD(0) update $V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ is an example of:
   A. Full sample updates.
   B. Bootstrapping.
   C. Exhaustive search.
   D. Offline learning.

   **Answer: B.** Bootstrapping. It updates $V(S_t)$ based on the estimate $V(S_{t+1})$.

9. In the Random Walk example, which method typically reaches a lower RMS error faster?
   A. MC
   B. TD(0)
   C. Dynamic Programming
   D. Random search

   **Answer: B.** TD(0). TD methods typically leverage the Markov property more efficiently through bootstrapping.

10. Sarsa is an **on-policy** control method because:
    A. It updates $Q(S_t, A_t)$ based on the action $A_{t+1}$ actually taken by the current policy.
    B. It uses the maximum $Q$-value of the next state for its update.
    C. It requires a model to predict the next state.
    D. It ignores exploration and follows a greedy policy.

    **Answer: A.** Updates based on action actually taken. Sarsa is (S, A, R, S', A').

11. Q-learning is an **off-policy** control method because:
    A. It follows a behavior policy but learns the optimal action-value function $q^*$.
    B. It always takes the most expensive action.
    C. it requires two separate behavior policies.
    D. It only works for episodic tasks.

    **Answer: A.** Learns optimal $q^*$ regardless of behavior. The update uses $\max_a Q(S', a)$, which decouples learning from exploration.

12. In the Cliff Walking task, why does Sarsa learn a "safer" path than Q-learning?
    A. Sarsa is less computationally expensive.
    B. Sarsa accounts for the exploration it actually does ($\epsilon$-greedy).
    C. Q-learning is biased toward lower rewards.
    D. Sarsa ignores the reward signal.

    **Answer: B.** Sarsa accounts for exploration. Sarsa realizes it might fall off the cliff due to $\epsilon$-greedy exploration and stays away.

13. Expected Sarsa reduces variance by:
    A. Updating all states in every step.
    B. Using the expected value over all next actions under the current policy.
    C. Eliminating the discount factor $\gamma$.
    D. Waiting until the end of the episode.

    **Answer: B.** Uses expected value over next actions. This averages out the noise from $A_{t+1}$ selection.

14. "Maximization Bias" in Q-learning refers to:
    A. The tendency to pick actions with high indices.
    B. Overestimating values due to taking a maximum over noisy estimates.
    C. The policy becoming too deterministic too quickly.
    D. The reward signal being scaled too high.

    **Answer: B.** Overestimating max over noisy estimates. $\max$ is a convex operator that favors high noise.

15. Double Q-learning resolves maximization bias by:
    A. Doubling the step-size $\alpha$.
    B. Using two independent estimates $Q_1$ and $Q_2$ to decouple selection from evaluation.
    C. Running two episodes simultaneously.
    D. Discounting rewards twice.

    **Answer: B.** Uses two independent estimates. One for $argmax$ (selection) and one for value (evaluation).
