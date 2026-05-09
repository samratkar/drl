---
layout: post
---

# Solutions - Subjective Questions

1. **Agent-Environment Boundary**: The boundary is not physical but functional. It separates everything the agent can change arbitrarily (the agent) from everything it cannot (the environment). Example: A robot's motor or its battery level. While they are part of the "robot," the agent (the decision logic) cannot arbitrarily set the battery level or the motor torque without interacting with the physics of the world.

2. **Markov Property**: Defined as $Pr\{S_{t+1}, R_{t+1} \mid S_t, A_t\} = Pr\{S_{t+1}, R_{t+1} \mid S_t, A_t, S_{t-1}, \ldots, S_0\}$. It means the current state and action provide all the necessary information to predict the future. It's important because it allows the state to be the sole basis for making decisions, ignoring the history.

3. **Dynamics Function**: $p(s', r \mid s, a) = Pr\{S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a\}$. It defines the probability of every possible next state and reward for every state-action pair. It describes the "rules of the game" (e.g., transition probabilities in a gridworld).

4. **Returns & Discounting**: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$. If $R_t$ is bounded and $\gamma < 1$, the sum is a convergent geometric series. $\gamma$ determines the "effective horizon": $\gamma \approx 0$ makes the agent "myopic" (short-term), while $\gamma \approx 1$ makes it "farsighted."

5. **State-Value vs. Action-Value**: $v_\pi(s)$ is the value of being in a state. $q_\pi(s,a)$ is the value of taking a specific action in a state. We prefer $q$ when the model is unknown (model-free) because it allows picking the best action without knowing the transition probabilities.

6. **Bellman Equation Intuition**: It states that the value of the current state is the expected reward plus the discounted value of the next state. It's "recursive consistency" because $v(s)$ is defined in terms of $v(s')$.

7. **Optimal Value Functions**: $v^*(s) = \max_\pi v_\pi(s)$ and $q^*(s,a) = \max_\pi q_\pi(s,a)$. The relationship is $v^*(s) = \max_a q^*(s,a)$.

8. **Bellman Optimality Equation**: $v^*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v^*(s')]$. It's non-linear because of the $\max$ operator, which makes it harder to solve than the linear Bellman equation for a fixed policy.

9. **Policy Improvement Theorem**: It states that if $q_\pi(s, \pi'(s)) \ge v_\pi(s)$ for all $s$, then policy $\pi'$ is at least as good as $\pi$. This justifies the greedy improvement step in DP.

10. **Policy Iteration Workflow**:
    - **Initialization**: Random $V(s)$ and $\pi(s)$.
    - **Evaluation**: Iterate $V(s) \leftarrow \sum_{s',r} p(s',r \mid s, \pi(s))[r + \gamma V(s')]$ until convergence.
    - **Improvement**: For each $s$, set $\pi(s) \leftarrow \arg\max_a \sum_{s',r} p(s',r \mid s, a)[r + \gamma V(s')]$.

11. **Value Iteration Mechanics**: It collapses evaluation and improvement: $V(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$. It only does one sweep of evaluation per "improvement" step, whereas PI does full evaluation.

12. **Generalized Policy Iteration (GPI)**: Evaluation makes the value function consistent with the current policy. Improvement makes the policy greedy with respect to the value function. They compete because changing one invalidates the other, but they cooperate to move both toward optimality ($v^*$ and $\pi^*$).

13. **Asynchronous DP**: These update states in any order and don't require full sweeps. They allow focusing computation on important states (e.g., those the agent actually visits) and can be more efficient for large state spaces.

14. **Bootstrapping**: Updating an estimate based on other estimates. In DP, $V(s)$ is updated using the current value of $V(s')$, which is also an estimate.

15. **Jack's Car Rental**:
    - **States**: Number of cars at two locations.
    - **Actions**: Moving cars between locations.
    - **Rewards**: Profit from rentals minus costs of moving cars.
    - **Poisson**: Rental requests and returns follow a Poisson distribution, requiring summation over these probabilities for the dynamics.

16. **Gambler's Problem**: All rewards are 0 until the goal is hit (+1). This creates a "sparse reward" environment where the value of a state reflects the probability of eventually hitting the goal from that state.

17. **Gridworld Boundary**: Usually, moving off the grid results in the agent staying in the same state and receiving a negative reward (e.g., -1). This ensures the state remains in the defined set.

18. **Unifying Episodic and Continuing**: We treat the end of an episode as a transition to a special "absorbing state" that transitions only to itself with reward 0 forever. This allows us to use the infinite sum formula for episodic tasks as well.

19. **Policy vs. Value**: No, an optimal policy MUST be greedy with respect to $v^*$. If it weren't, there would be a better action (higher $q^*(s,a)$), contradicting the optimality of the policy.

20. **Complexity of DP**: DP is polynomial in $\mid S \mid$ and $\mid A \mid$. However, the number of states $\mid S \mid$ often grows exponentially with the number of state variables (e.g., a grid of $10\times10$ is 100 states, but two such grids is $100^2$ states). This is the "curse of dimensionality."
of dimensionality."
