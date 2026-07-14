---
layout: post
---

# MCQ Solutions - Lecture 2

1. The Multi-Armed Bandit problem is "non-associative" because:
   A. Actions are not associated with rewards.
   B. Actions do not depend on different states.
   C. The reward distribution is always non-stationary.
   D. There is no optimal action.

   **Answer: B.** Actions do not depend on states. Non-associative means the agent doesn't need to associate actions with different state signals.

2. In the k-armed bandit problem, "Exploitation" refers to:
   A. Picking an action at random to gain information.
   B. Picking the action with the highest current value estimate.
   C. Picking an action that has never been tried before.
   D. Changing the reward distribution of the environment.

   **Answer: B.** Highest current value estimate. Exploitation is choosing what is currently believed to be the best.

3. The $\epsilon$-greedy method selects a random action with probability:
   A. $1 - \epsilon$
   B. $\epsilon$
   C. $\epsilon / k$
   D. $1 / k$

   **Answer: B.** $\epsilon$. In an $\epsilon$-greedy step, with probability $\epsilon$ a random action is chosen.

4. Which of the following methods uses an "exploration bonus" based on uncertainty?
   A. Greedy
   B. $\epsilon$-greedy
   |C. Upper-Confidence-Bound (UCB)
   D. Gradient Bandit

   **Answer: C.** UCB. It adds $c \sqrt{\ln t / N_t(a)}$ as a bonus.

5. Gradient Bandit algorithms select actions based on:
   A. Sample averages of rewards.
   B. Numerical preferences $H_t(a)$ via a softmax distribution.
   C. Upper confidence bounds.
   D. The total number of steps $t$.

   **Answer: B.** Preferences via softmax. It learns $H_t(a)$ and uses softmax for $\pi_t(a)$.

6. In non-stationary bandit problems, it is better to use:
   A. Sample-average updates ($1/n$).
   B. Constant step-size updates ($\alpha$).
   C. A purely greedy policy.
   D. No exploration at all.

   **Answer: B.** Constant step-size. It gives more weight to recent rewards (forgetting old ones).

7. Optimistic Initial Values ($Q_1 = +5$) encourage:
   A. Early exploitation of the first arm tried.
   B. Systematic early exploration through disappointment.
   C. Faster convergence to a suboptimal arm.
   D. The use of a baseline in gradient updates.

   **Answer: B.** Systematic early exploration. Everything looks disappointing initially, forcing the agent to try everything.

8. The UCB formula $A_t = \argmax_a [Q_t(a) + c \sqrt{\ln t / N_t(a)}]$ becomes purely greedy as:
   A. $t \to \infty$ and $N_t(a)$ is large.
   B. $c \to \infty$.
   C. $Q_t(a)$ becomes zero.
   D. $\ln t$ becomes zero.

   **Answer: A.** $t \to \infty$. As $N_t(a)$ grows, the uncertainty term shrinks to zero.

9. In Gradient Bandit algorithms, the "baseline" (average reward) is used to:
   A. Change the optimal action.
   B. Reduce the variance of the preference updates.
   C. Increase the exploration rate.
   D. Eliminate the need for $\alpha$.

   **Answer: B.** Reduce variance. It compares the reward to the average, making updates more stable.

10. A step-size sequence $\alpha_n$ is guaranteed to converge to the true value if:
    A. $\sum \alpha_n = \infty$ and $\sum \alpha_n^2 < \infty$.
    B. $\alpha_n$ is constant.
    C. $\alpha_n = 1$.
    D. $\sum \alpha_n < \infty$.

    **Answer: A.** $\sum \alpha_n = \infty$ (large enough) and $\sum \alpha_n^2 < \infty$ (settles down).
