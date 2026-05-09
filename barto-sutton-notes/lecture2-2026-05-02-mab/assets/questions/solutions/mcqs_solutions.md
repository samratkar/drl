# MCQ Solutions - Lecture 2

1. **Answer: B.** Actions do not depend on states. Non-associative means the agent doesn't need to associate actions with different state signals.
2. **Answer: B.** Highest current value estimate. Exploitation is choosing what is currently believed to be the best.
3. **Answer: B.** $\epsilon$. In an $\epsilon$-greedy step, with probability $\epsilon$ a random action is chosen.
4. **Answer: C.** UCB. It adds $c \sqrt{\ln t / N_t(a)}$ as a bonus.
5. **Answer: B.** Preferences via softmax. It learns $H_t(a)$ and uses softmax for $\pi_t(a)$.
6. **Answer: B.** Constant step-size. It gives more weight to recent rewards (forgetting old ones).
7. **Answer: B.** Systematic early exploration. Everything looks disappointing initially, forcing the agent to try everything.
8. **Answer: A.** $t \to \infty$. As $N_t(a)$ grows, the uncertainty term shrinks to zero.
9. **Answer: B.** Reduce variance. It compares the reward to the average, making updates more stable.
10. **Answer: A.** $\sum \alpha_n = \infty$ (large enough) and $\sum \alpha_n^2 < \infty$ (settles down).
