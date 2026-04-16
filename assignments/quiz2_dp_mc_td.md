# Quiz 2: Dynamic Programming, Monte Carlo, and Temporal Difference Methods

**Course:** Deep Reinforcement Learning (MTech)
**Total Marks:** 5 (20 questions x 0.25 marks each)
**Duration:** 30 minutes
**Reference:** Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (Chapters 4, 5, 6)

---

## Section A: Dynamic Programming [7 questions, 1.75 marks]

**Q1.** The Bellman optimality equation for the state-value function is:

$$v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma\, v_*(s')\right]$$

Which of the following is a fundamental assumption required for this equation to be solved exactly using dynamic programming?

(a) The agent must have access to a simulator of the environment
(b) The agent must have complete knowledge of the environment's dynamics $p(s',r|s,a)$
(c) The state space must be continuous so that derivatives can be computed
(d) The reward function must be deterministic

---

**Q2.** In policy evaluation, the Bellman expectation equation is:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\left[r + \gamma\, V^\pi(s')\right]$$

The same $V^\pi$ appears on both sides — this is a fixed-point equation defining a system of simultaneous equations over all states. To solve it, we initialize $V(s) = 0$ for all states and iteratively apply the equation. What guarantees that this iterative process converges to the true $V^\pi$?

(a) The policy $\pi$ must be deterministic for convergence
(b) The Bellman operator is a **contraction mapping** under the discount factor $\gamma < 1$, so repeated application from any initial guess converges to the unique fixed point $V^\pi$
(c) Convergence is only guaranteed if the initial guess $V_0$ is close to $V^\pi$
(d) The method converges only for environments with deterministic transitions

---

**Q3.** Consider a $4 \times 4$ gridworld where all transitions yield a reward of $-1$ and the discount factor $\gamma = 1$. Under the equiprobable random policy, after running policy evaluation to convergence, which cells will have the **most negative** state values?

(a) Cells adjacent to a terminal state
(b) Cells in the center of the grid, farthest from any terminal state
(c) Cells along the boundary of the grid
(d) All non-terminal cells will have the same value

---

**Q4.** **Policy iteration** alternates between policy evaluation and policy improvement. The policy improvement theorem guarantees that if $q_\pi(s, \pi'(s)) \geq v_\pi(s)$ for all states $s$, then:

(a) $\pi'$ is the optimal policy
(b) $v_{\pi'} = v_{\pi}$ for all states
(c) $v_{\pi'}(s) \geq v_{\pi}(s)$ for all states $s$
(d) The Q-values under $\pi'$ are unchanged

---

**Q5.** **Value iteration** can be understood as:

(a) Policy evaluation run for exactly one sweep, followed by policy improvement, repeated
(b) Policy evaluation run to convergence, followed by a single policy improvement step
(c) Applying the Bellman expectation equation as an update rule
(d) A Monte Carlo method applied to the Bellman equation

---

**Q6.** Generalized Policy Iteration (GPI) refers to the general idea of interleaving policy evaluation and policy improvement. Which of the following statements about GPI is **FALSE**?

(a) Policy iteration and value iteration are both special cases of GPI
(b) GPI requires that policy evaluation be run to convergence before each improvement step
(c) The evaluation and improvement processes compete in the sense that making the policy greedy w.r.t. the value function changes the value function
(d) If both evaluation and improvement stabilize, the resulting policy and value function are optimal

---

**Q7.** A major limitation of dynamic programming methods is the "curse of dimensionality." For an environment with $n$ state variables each taking $k$ possible values and $m$ actions, what is the computational complexity per full sweep of value iteration?

(a) $O(n \cdot k \cdot m)$
(b) $O(k^n \cdot m)$
(c) $O(k^n \cdot m \cdot k^n)$ since each state transition can lead to any of $k^n$ next states
(d) $O(n^k \cdot m)$

---

## Section B: Monte Carlo Methods [8 questions, 2.0 marks]

**Q8.** Monte Carlo (MC) methods estimate value functions by averaging **returns** from sampled episodes. Unlike dynamic programming, MC methods do NOT require:

(a) Knowledge of the reward function
(b) A model of the environment's transition dynamics
(c) The discount factor $\gamma$ to be less than 1
(d) The policy to be stationary

---

**Q9.** In **first-visit MC prediction**, the value of a state $s$ is estimated by averaging the returns following:

(a) Every occurrence of $s$ in every episode
(b) Only the first time $s$ is visited within each episode
(c) The last occurrence of $s$ in every episode
(d) A randomly selected visit to $s$ in each episode

Which variant (first-visit vs. every-visit) has an unbiased estimate of $v_\pi(s)$?

(a) Only first-visit MC
(b) Only every-visit MC
(c) Both are unbiased
(d) Neither is unbiased

---

**Q10.** In on-policy MC control, we use an $\varepsilon$-greedy policy. After running MC control to convergence, the resulting policy is optimal among:

(a) All possible policies
(b) All deterministic policies
(c) All $\varepsilon$-soft policies (policies that assign at least $\varepsilon/|\mathcal{A}|$ to every action)
(d) All stochastic policies with entropy above a threshold

---

**Q11.** Why is it important to maintain **exploring starts** or use an $\varepsilon$-soft policy in Monte Carlo control?

(a) To ensure the value function converges faster
(b) To guarantee that every state-action pair is visited infinitely often in the limit, which is necessary for convergence to optimal values
(c) To reduce the variance of the return estimates
(d) To ensure that episodes terminate in finite time

---

**Q12.** In **off-policy Monte Carlo prediction** using **ordinary importance sampling**, the ratio for a trajectory from time $t$ to the terminal time $T$ is:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

What is a well-known problem with ordinary importance sampling?

(a) It is biased but has low variance
(b) It is unbiased but can have very high (potentially infinite) variance
(c) It requires the behavior policy $b$ to be deterministic
(d) It can only be used when $\pi$ and $b$ are close to each other

---

**Q13.** **Weighted importance sampling** differs from ordinary importance sampling in that:

(a) It uses a weighted average of returns where each return is weighted by the cumulative importance-sampling ratio
(b) It only weights the first return from each episode
(c) It eliminates the need for importance-sampling ratios altogether
(d) It requires the target policy to be $\varepsilon$-greedy

What is the key statistical trade-off compared to ordinary importance sampling?

(a) Higher variance, lower bias
(b) Lower variance, but biased (though bias asymptotically approaches zero)
(c) Both bias and variance are higher
(d) Both bias and variance are lower

---

**Q14.** In off-policy MC control, the **coverage assumption** requires that:

(a) The target policy $\pi$ must be a subset of the behavior policy $b$
(b) $\pi(a|s) > 0$ implies $b(a|s) > 0$ for all $s$ and $a$
(c) $b(a|s) > 0$ implies $\pi(a|s) > 0$ for all $s$ and $a$
(d) Both $\pi$ and $b$ must be $\varepsilon$-soft

---

**Q15.** A robot learning to navigate a maze uses MC methods. After 1000 episodes, some states near the goal have been visited 500 times, while some states in distant corners have been visited only 3 times. Which statement is most accurate?

(a) The value estimates for all states are equally reliable since MC is unbiased
(b) The value estimates for rarely-visited states will have much higher variance and should be treated with less confidence
(c) MC methods automatically adjust the learning rate for rarely-visited states, so this imbalance does not matter
(d) The agent should restart training with a uniform-visit policy

---

## Section C: Temporal Difference Methods [5 questions, 1.25 marks]

**Q16.** The TD(0) update rule is:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma\, V(S_{t+1}) - V(S_t) \right]$$

The quantity $\delta_t = R_{t+1} + \gamma\, V(S_{t+1}) - V(S_t)$ is called the **TD error**. TD(0) combines ideas from both MC and DP. Specifically, TD(0) is similar to MC in that it ______ and similar to DP in that it ______.

(a) learns from complete episodes; requires a model of the environment
(b) uses sampling (does not require a model); bootstraps (updates estimates based on other estimates)
(c) is model-free; uses the full Bellman equation for updates
(d) averages returns from episodes; sweeps over the entire state space

---

**Q17.** Consider the following two sequences of experience under the same policy:

- **Episode 1:** A, 0, B, 0, (terminal)
- **Episode 2:** B, 1, (terminal)
- **Episode 3:** B, 1, (terminal)
- **Episode 4:** B, 1, (terminal)
- **Episode 5:** B, 1, (terminal)
- **Episode 6:** B, 1, (terminal)
- **Episode 7:** B, 0, (terminal)
- **Episode 8:** B, 1, (terminal)

With $\gamma = 1$, what are the TD(0) batch estimates (trained to convergence) for $V(A)$ and $V(B)$?

(a) $V(A) = 0$, $V(B) = 0.75$
(b) $V(A) = 0.75$, $V(B) = 0.75$
(c) $V(A) = 0$, $V(B) = 0.875$
(d) $V(A) = 0.75$, $V(B) = 0.875$

---

**Q18.** **SARSA** is an on-policy TD control method with the update:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

**Q-learning** is an off-policy TD control method with the update:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma\, \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

In the classic **cliff walking** environment with an $\varepsilon$-greedy policy, which statement is correct?

(a) Q-learning finds the optimal path (along the cliff edge) but obtains higher online reward than SARSA
(b) SARSA learns the safer path (away from the cliff) and typically obtains higher online reward because it accounts for its own exploratory actions
(c) Both algorithms find the same path because they have the same convergence guarantee
(d) SARSA converges faster than Q-learning in all environments

---

**Q19.** **Expected SARSA** uses the update:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})\, Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

What advantage does Expected SARSA have over standard SARSA?

(a) It eliminates the need for an exploration policy
(b) It reduces variance in the update by taking an expectation over the next action instead of sampling a single action
(c) It converges to the optimal policy even without exploration
(d) It requires fewer episodes to learn because it uses a model

---

**Q20.** **Maximization bias** is a problem in Q-learning because:

(a) The discount factor $\gamma$ causes future rewards to be systematically overvalued
(b) The $\max$ operator over estimated Q-values introduces a positive bias, as the maximum of noisy estimates tends to overestimate the true maximum
(c) The learning rate $\alpha$ causes values to grow unboundedly
(d) The $\varepsilon$-greedy policy selects suboptimal actions too often

**Double Q-learning** addresses this by:

(a) Maintaining two independent Q-value estimates and using one to select the best action and the other to evaluate it
(b) Using the average of two Q-tables for action selection
(c) Halving the learning rate
(d) Using importance sampling to correct the bias

---

## Answer Key (For Instructor Use Only)

| Q  | Answer |
|----|--------|
| 1  | (b)    |
| 2  | (b)    |
| 3  | (b)    |
| 4  | (c)    |
| 5  | (a)    |
| 6  | (b)    |
| 7  | (c)    |
| 8  | (b)    |
| 9  | (b), (c) |
| 10 | (c)    |
| 11 | (b)    |
| 12 | (b)    |
| 13 | (a), (b) |
| 14 | (b)    |
| 15 | (b)    |
| 16 | (b)    |
| 17 | (b)    |
| 18 | (b)    |
| 19 | (b)    |
| 20 | (b), (a) |

---

## Brief Answer Justifications

**Q1 (b):** DP requires the full model $p(s',r|s,a)$ to compute the expected updates over all possible next states and rewards.

**Q2 (b):** The Bellman operator is a contraction mapping with contraction factor $\gamma$. This means each iterative application brings the estimate closer to the unique fixed point $V^\pi$, regardless of the initial guess. The contraction mapping theorem (Banach fixed-point theorem) guarantees both existence of a unique solution and convergence of the iterative procedure.

**Q3 (b):** With $\gamma=1$ and reward $-1$ per step, the value of a state equals the negative expected number of steps to reach the terminal state. Center cells are farthest from any terminal state under the random policy.

**Q4 (c):** The policy improvement theorem states that the new greedy policy $\pi'$ is at least as good as $\pi$ everywhere; it does not guarantee optimality in one step.

**Q5 (a):** Value iteration truncates policy evaluation to a single sweep (one backup per state) and immediately performs a greedy improvement. This is equivalent to directly applying the Bellman optimality operator.

**Q6 (b):** GPI does NOT require full convergence of policy evaluation before improvement. Value iteration (one-sweep evaluation) and asynchronous DP are both valid forms of GPI.

**Q7 (c):** Each of the $k^n$ states requires summing over $m$ actions, and for each action the transition model sums over all $k^n$ possible next states, giving $O(k^n \cdot m \cdot k^n)$ per sweep.

**Q8 (b):** MC methods learn directly from sampled episodes of experience and do not need a model of the environment's transition probabilities.

**Q9 (b), (c):** First-visit MC averages returns from only the first visit. Both first-visit and every-visit MC produce unbiased estimates of $v_\pi(s)$ (every-visit MC's unbiasedness follows from the fact that each visit's return is an unbiased sample).

**Q10 (c):** On-policy MC with $\varepsilon$-greedy finds the best policy within the class of $\varepsilon$-soft policies, not the globally optimal policy.

**Q11 (b):** Without sufficient exploration, some state-action pairs may never be visited, preventing convergence to optimal values. Exploring starts or $\varepsilon$-soft policies ensure all pairs are visited infinitely often.

**Q12 (b):** Ordinary importance sampling is unbiased but a single large importance-sampling ratio can cause the variance to explode, especially with long episodes.

**Q13 (a), (b):** Weighted importance sampling uses a weighted average with the ratios as weights. It has lower variance than ordinary IS but introduces a small bias that vanishes asymptotically.

**Q14 (b):** Coverage requires that every action the target policy might take must also have nonzero probability under the behavior policy ($\pi(a|s)>0 \Rightarrow b(a|s)>0$).

**Q15 (b):** While MC estimates are unbiased, the variance of the estimate decreases with more samples. States visited only 3 times will have much higher variance than those visited 500 times.

**Q16 (b):** TD is model-free like MC (it samples transitions) and bootstraps like DP (it updates estimates using other estimates rather than waiting for complete returns).

**Q17 (b):** Batch TD(0) converges to the maximum-likelihood MDP model. $V(B) = 6/8 = 0.75$. Since A transitions to B with reward 0, $V(A) = 0 + \gamma \cdot V(B) = 0.75$.

**Q18 (b):** SARSA's on-policy nature means it accounts for the exploratory $\varepsilon$-greedy actions, learning a safer path away from the cliff. This yields higher online reward despite the path being suboptimal under a greedy policy. Q-learning learns the optimal (cliff-edge) path but suffers lower online reward from occasional exploratory falls.

**Q19 (b):** Expected SARSA replaces the single sampled next action with an expectation over all actions, eliminating variance due to action selection while maintaining the same expected update.

**Q20 (b), (a):** The max over noisy estimates is a positively biased estimator of the true max. Double Q-learning decouples action selection from evaluation using two independent estimators, which removes this bias.
