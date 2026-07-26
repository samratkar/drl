---
layout: post
---

# MCQ Solutions (Lecture 11: Combined, Advanced PG & Model-Based RL)

1. **Answer: b**
   * **Explanation:** TRPO guarantees monotonic policy improvement by enforcing a KL-divergence constraint on policy updates ($\mathbb{E}[D_{KL}(\pi_{old} || \pi)] \le \delta$), ensuring the new policy does not deviate too far from the old policy's distribution.

2. **Answer: b**
   * **Explanation:** Model-based RL methods learn/use a model of environment transitions, allowing them to plan offline with "imagined" data (high sample-efficiency). However, running simulations or tree searches at decision time is computationally expensive.

3. **Answer: c**
   * **Explanation:** In the simulation (rollout) step of MCTS, actions are chosen using a fast, default rollout policy (which can be fully random) to quickly reach a terminal state and determine a reward value.

4. **Answer: c**
   * **Explanation:** AlphaGo evaluates leaf nodes by combining both a state evaluation from the Value Network ($v_{\theta}$) and a rollout outcome ($z$) from the fast rollout policy, using a mixing weight $\lambda$ (specifically $\lambda = 0.5$).

5. **Answer: b**
   * **Explanation:** The Dynamics Function ($g_{\theta}$) in MuZero computes the next latent state $s^k$ and the immediate reward $r^k$ from the previous latent state $s^{k-1}$ and action $a_k$.

6. **Answer: a**
   * **Explanation:** Behavior Cloning only learns from expert trajectories. When the agent makes a small mistake (covariate shift), it lands in state distributions it has never seen, leading to poor actions that compound over time and cause the agent to fail.

7. **Answer: a**
   * **Explanation:** GAIL trains a discriminator $D_{\phi}(s, a)$ to output the probability that a state-action pair is from the expert. The surrogate reward maximized by the policy is $R(s, a) = -\ln(1 - D_{\phi}(s, a))$. As the agent gets better at mimicking the expert, $D_{\phi}(s, a) \rightarrow 1$, which makes the reward highly positive.


