# MCQ Solutions - Lecture 4

1. **Answer: C.** Learning from complete episodes. MC methods wait for the episode to end before updating estimates.
2. **Answer: B.** The first time $s$ is visited in each episode. Every-visit MC (C) would average all visits.
3. **Answer: B.** Ensure every state-action pair is visited. This solves the exploration problem without using stochastic policies.
4. **Answer: B.** The same policy used for decisions. On-policy learning is "self-evaluating."
5. **Answer: A.** Target actions must be possible under behavior policy. You cannot learn about an action if you never see it.
6. **Answer: B.** Weighted Importance Sampling. It has lower variance than ordinary IS, although it is slightly biased (toward 0).
7. **Answer: A.** MC and Dynamic Programming. TD learns from experience (like MC) and bootstraps (like DP).
8. **Answer: B.** Bootstrapping. It updates $V(S_t)$ based on the estimate $V(S_{t+1})$.
9. **Answer: B.** TD(0). TD methods typically leverage the Markov property more efficiently through bootstrapping.
10. **Answer: A.** Updates based on action actually taken. Sarsa is (S, A, R, S', A').
11. **Answer: A.** Learns optimal $q^*$ regardless of behavior. The update uses $\max_a Q(S', a)$, which decouples learning from exploration.
12. **Answer: B.** Sarsa accounts for exploration. Sarsa realizes it might fall off the cliff due to $\epsilon$-greedy exploration and stays away.
13. **Answer: B.** Uses expected value over next actions. This averages out the noise from $A_{t+1}$ selection.
14. **Answer: B.** Overestimating max over noisy estimates. $\max$ is a convex operator that favors high noise.
15. **Answer: B.** Uses two independent estimates. One for $argmax$ (selection) and one for value (evaluation).
