# Solutions - MCQs Set 1

1. **B**. The sequence is $S_0, A_0, R_1, S_1, A_1, R_2, ...$. State, then Action, then Reward and next State.
2. **B**. The reward signal generation is outside the agent's arbitrary control, thus part of the environment.
3. **B**. This is the formal definition of the Markov property, where the current state and action are sufficient to predict the next state and reward.
4. **A**. It is a probability distribution over the next state and reward given the current state and action.
5. **A**. By summing over all possible rewards, we get the marginal probability of the next state.
6. **A**. The expected reward is the sum of rewards weighted by their probabilities $p(s', r | s, a)$.
7. **B**. With $\gamma = 0$, $G_t = R_{t+1}$, so only the immediate reward matters.
8. **A**. $G_t = \sum_{k=0}^{\infty} \gamma^k (1) = 1/(1-\gamma)$ (geometric series).
9. **A**. It relates the value of a state to the expected values of its possible successor states.
10. **A**. This is the standard Bellman equation for $v_\pi$.
11. **A**. $q^*(s, a)$ is the maximum action-value achievable by any policy.
12. **C**. The Bellman optimality equation uses the $\max$ operator to pick the best action.
13. **A**. Policy Iteration alternates between evaluating the current policy and improving it greedily.
14. **C**. Convergence is typically measured by the change in the value function being below a threshold $\theta$.
15. **A**. Value iteration essentially does one sweep of evaluation and then improves.
16. **A**. GPI is the conceptual framework of two processes (evaluation and improvement) moving toward each other.
17. **A**. The "sawtooth" occurs because for certain capital amounts, a large bet can reach the goal in one step, while for others, multiple steps are needed.
18. **A**. Asynchronous DP does not require a systematic sweep of the entire state space.
19. **A**. RL methods bootstrap when they use estimates to update other estimates (like $V(s')$ to update $V(s)$).
20. **B**. DP is polynomial in $|S|$ and $|A|$, which is much better than the exponential complexity of exhaustive search.
