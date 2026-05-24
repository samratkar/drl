---
layout: post
---

# Solutions - MCQs Set 1

1. In a finite MDP, which of the following best describes the sequence of random variables starting at time $t=0$?
   A. $S_0, R_1, A_0, S_1, R_2, A_1, ...$
   B. $S_0, A_0, R_1, S_1, A_1, R_2, ...$
   C. $A_0, S_0, R_1, A_1, S_1, R_2, ...$
   D. $R_1, S_0, A_0, R_2, S_1, A_1, ...$

   **Answer: B.** The sequence is $S_0, A_0, R_1, S_1, A_1, R_2, ...$. State, then Action, then Reward and next State.

2. According to the "Agent-Environment Boundary," which of the following is considered part of the environment?
   A. The agent's decision-making policy.
   B. The internal reward signal generation process.
   C. The specific algorithm used for value estimation.
   D. The agent's chosen action $A_t$.

   **Answer: B.** The reward signal generation is outside the agent's arbitrary control, thus part of the environment.

3. A state signal $S_t$ possesses the Markov property if and only if:
   A. $Pr\{S_{t+1} = s' \mid S_t = s\} = Pr\{S_{t+1} = s' \mid S_t = s, A_t = a\}$
   B. $Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t, A_t\} = Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0\}$
   C. The future is independent of the present given the past.
   D. The transitions are deterministic.

   **Answer: B.** This is the formal definition of the Markov property, where the current state and action are sufficient to predict the next state and reward.

4. The dynamics function $p(s', r \mid s, a)$ defines a probability distribution over $s'$ and $r$ such that:
   A. $\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1$ for all $s, a$.
   B. $\sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} p(s', r \mid s, a) = 1$ for all $s', r$.
   C. $p(s', r \mid s, a) = p(s' \mid s, a) + p(r \mid s, a)$.
   D. It represents the probability of action $a$ being taken in state $s$.

   **Answer: A.** It is a probability distribution over the next state and reward given the current state and action.

5. How is the state-transition probability $p(s' \mid s, a)$ derived from the dynamics function $p(s', r \mid s, a)$?
   A. $\sum_{r \in \mathcal{R}} p(s', r \mid s, a)$
   B. $\max_{r \in \mathcal{R}} p(s', r \mid s, a)$
   C. $\prod_{r \in \mathcal{R}} p(s', r \mid s, a)$
   D. $p(s', r \mid s, a) / r$

   **Answer: A.** By summing over all possible rewards, we get the marginal probability of the next state.

6. The expected reward for state-action pair $(s, a)$, denoted $r(s, a)$, is calculated as:
   A. $\sum_{s', r} r \cdot p(s', r \mid s, a)$
   B. $\sum_{s'} p(s' \mid s, a)$
   C. $\max_{s', r} r$
   D. $\sum_{r} p(r \mid s, a)$

   **Answer: A.** The expected reward is the sum of rewards weighted by their probabilities $p(s', r \mid s, a)$.

7. In the context of returns, if $\gamma = 0$, the agent is:
   A. Far-sighted, caring about all future rewards equally.
   B. Only interested in maximizing the immediate reward $R_{t+1}$.
   C. Ignoring all rewards, including the immediate one.
   D. Following a random policy.

   **Answer: B.** With $\gamma = 0$, $G_t = R_{t+1}$, so only the immediate reward matters.

8. For a continuing task with constant reward $R_t = 1$ for all $t$, what is the return $G_t$ if $0 < \gamma < 1$?
   A. $1 / (1 - \gamma)$
   B. $1 / \gamma$
   C. $\gamma / (1 - \gamma)$
   D. $\infty$

   **Answer: A.** $G_t = \sum_{k=0}^{\infty} \gamma^k (1) = 1/(1-\gamma)$ (geometric series).

9. The Bellman equation for $v_\pi(s)$ expresses a relationship between:
   A. The value of a state and the value of its successor states.
   B. The value of a state and the optimal value $v^*(s)$.
   C. The reward $R_{t+1}$ and the action $A_t$.
   D. Two different policies $\pi$ and $\pi'$.

   **Answer: A.** It relates the value of a state to the expected values of its possible successor states.

10. Which equation correctly represents the Bellman equation for $v_\pi(s)$?
    A. $v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_\pi(s')]$
    B. $v_\pi(s) = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_\pi(s')]$
    C. $v_\pi(s) = \sum_a \pi(a \mid s) [R_{t+1} + \gamma v_\pi(S_{t+1})]$
    D. $v_\pi(s) = \mathbb{E}[G_t \mid S_t = s, A_t = a]$

    **Answer: A.** This is the standard Bellman equation for $v_\pi$.

11. The optimal action-value function $q^*(s, a)$ is defined as:
    A. $\max_\pi q_\pi(s, a)$ for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$.
    B. $\sum_\pi q_\pi(s, a)$.
    C. $q_\pi(s, a)$ where $\pi$ is the uniform random policy.
    D. The average of all possible action-values.

    **Answer: A.** $q^*(s, a)$ is the maximum action-value achievable by any policy.

12. The Bellman optimality equation for $v^*(s)$ uses which operator to select actions?
    A. Summation ($\sum$)
    B. Expectation ($\mathbb{E}$)
    C. Maximum ($\max$)
    D. Minimum ($\min$)

    **Answer: C.** The Bellman optimality equation uses the $\max$ operator to pick the best action.

13. Policy Iteration consists of two alternating phases. What are they?
    A. Policy Evaluation and Policy Improvement.
    B. Value Iteration and Reward Maximization.
    C. Exploration and Exploitation.
    D. Action Selection and State Transition.

    **Answer: A.** Policy Iteration alternates between evaluating the current policy and improving it greedily.

14. During Policy Evaluation, the value function $V$ is updated until:
    A. $V$ matches $v^*$ exactly.
    B. The policy $\pi$ changes.
    C. The maximum change across all states is less than a small threshold $\theta$.
    D. 100 iterations are completed.

    **Answer: C.** Convergence is typically measured by the change in the value function being below a threshold $\theta$.

15. Value Iteration can be seen as a special case of Policy Evaluation where:
    A. Evaluation is stopped after just one sweep.
    B. The policy is always random.
    C. There is no discounting ($\gamma = 1$).
    D. Rewards are always positive.

    **Answer: A.** Value iteration essentially does one sweep of evaluation and then improves.

16. Generalized Policy Iteration (GPI) refers to:
    A. The general idea of letting evaluation and improvement processes interact.
    B. A specific algorithm that combines SARSA and Q-Learning.
    C. Using neural networks for policy approximation.
    D. The process of converting a continuing task to an episodic one.

    **Answer: A.** GPI is the conceptual framework of two processes (evaluation and improvement) moving toward each other.

17. In the Gambler's Problem, the state is the gambler's capital. What makes the optimal policy "sawtooth" in shape?
    A. The discrete nature of the bets and the goal.
    B. The use of a linear reward function.
    C. The assumption that $p=0.5$.
    D. The inclusion of a transaction fee for every bet.

    **Answer: A.** The "sawtooth" occurs because for certain capital amounts, a large bet can reach the goal in one step, while for others, multiple steps are needed.

18. Asynchronous Dynamic Programming algorithms are characterized by:
    A. Updating state values in any order, using whatever values are available.
    B. Requiring a full sweep of the state set before any value is updated.
    C. Running on multiple CPUs simultaneously.
    D. Only being applicable to episodic tasks.

    **Answer: A.** Asynchronous DP does not require a systematic sweep of the entire state space.

19. Bootstrapping in RL refers to:
    A. Updating a value estimate based on other value estimates.
    B. Starting the agent with a random policy.
    C. Using a replay buffer to store experiences.
    D. Normalizing rewards to a [0, 1] range.

    **Answer: A.** RL methods bootstrap when they use estimates to update other estimates (like $V(s')$ to update $V(s)$).

20. The computational complexity of DP methods is:
    A. Exponential in the number of states and actions.
    B. Polynomial in the number of states and actions.
    C. Independent of the number of states.
    D. Logarithmic in the number of actions.

    **Answer: B.** DP is polynomial in $|S|$ and $|A|$, which is much better than the exponential complexity of exhaustive search.
