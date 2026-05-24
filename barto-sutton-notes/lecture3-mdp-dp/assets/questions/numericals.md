---
layout: post
---

# Numerical Questions - MDPs & Dynamic Programming

1. **Return Calculation**: An agent receives a sequence of rewards: $R_1=2, R_2=-1, R_3=5$. The episode ends after $R_3$. Calculate the return $G_0$ assuming $\gamma = 1$.

2. **Discounted Return**: For the same sequence ($R_1=2, R_2=-1, R_3=5$), calculate $G_0$ with a discount factor $\gamma = 0.5$.

3. **Infinite Horizon Return**: A continuing task yields a constant reward $R_t = 10$ for all $t$. If $\gamma = 0.9$, what is the total return $G_t$?

4. **Expected Reward**: Given the dynamics $p(s_1, 5 \mid s, a) = 0.3$ and $p(s_2, -2 \mid s, a) = 0.7$, calculate the expected reward $r(s, a)$.

5. **State-Transition Probability**: Using the dynamics from question 4, what is the state-transition probability $p(s_1 \mid s, a)$?

6. **Bellman Equation (State Value)**: In a gridworld, state $S$ has four possible successor states with values $V(S_{up})=10, V(S_{down})=2, V(S_{left})=0, V(S_{right})=5$. If the policy $\pi$ is uniform random and $\gamma = 0.9$ (with $R=0$ for all moves), calculate $V(S)$.

7. **Bellman Equation (Action Value)**: Given $p(s', 10 \mid s, a) = 1.0$ and $V(s') = 20$, calculate $q_\pi(s, a)$ with $\gamma = 0.9$.

8. **Policy Evaluation Update**: A state $s$ has only one action $a$. $p(s, 1 \mid s, a) = 0.2$ and $p(s', 10 \mid s, a) = 0.8$. If current estimates are $V(s)=0$ and $V(s')=5$, what is the new estimate $V(s)$ after one update with $\gamma=1$?

9. **Value Iteration Update**: State $s$ has two actions, $A$ and $B$.
   - Action $A$: Leads to $s_1$ (value 10) with reward 0.
   - Action $B$: Leads to $s_2$ (value 5) with reward 6.
   Assume $\gamma = 0.9$. Calculate the new value $V(s)$ using the Value Iteration update.

10. **Greedy Action Selection**: Given $q(s, a_1) = 4.5, q(s, a_2) = 5.2, q(s, a_3) = 5.1$, which action is chosen by a greedy policy $\pi'(s)$?

11. **Gridworld Boundary Reward**: Consider a 3×3 gridworld with a uniform random policy ($\pi = 1/4$ for each action: N, S, E, W). The rules are:
    - Moving off the grid: agent stays in the same cell and receives $R = -1$.
    - Moving to any valid non-terminal cell: $R = 0$.
    - Moving into the goal state at $(2,2)$: $R = +10$, episode ends. $V(\text{terminal}) = 0$.
    - $\gamma = 0.9$.

    The converged state-value function $V(s)$ and action-value function $Q(s,a)$ under this policy are:

    **State Values $V(s)$:**
    ```
              Col 0    Col 1    Col 2
            +--------+--------+--------+
    Row 0   | -0.86  |  0.06  |  0.56  |
            +--------+--------+--------+
    Row 1   |  0.06  |  1.62  |  3.54  |
            +--------+--------+--------+
    Row 2   |  0.56  |  3.54  |  0.00  |  ← Goal (terminal)
            +--------+--------+--------+
    ```

    **Action Values $Q(s, a)$:**
    | State | $Q(s, N)$ | $Q(s, S)$ | $Q(s, E)$ | $Q(s, W)$ |
    |-------|-----------|-----------|-----------|-----------|
    | $(0,0)$ | $-1.77$ | $0.06$ | $0.06$ | $-1.77$ |
    | $(0,1)$ | $-0.94$ | $1.46$ | $0.51$ | $-0.77$ |
    | $(0,2)$ | $-0.49$ | $3.18$ | $-0.49$ | $0.06$ |
    | $(1,0)$ | $-0.77$ | $0.51$ | $1.46$ | $-0.94$ |
    | $(1,1)$ | $0.06$ | $3.18$ | $3.18$ | $0.06$ |
    | $(1,2)$ | $0.51$ | $10.00$ | $2.18$ | $1.46$ |
    | $(2,0)$ | $0.06$ | $-0.49$ | $3.18$ | $-0.49$ |
    | $(2,1)$ | $1.46$ | $2.18$ | $10.00$ | $0.51$ |

    (a) Draw the grid showing which actions hit boundaries for state $(0, 2)$.
    (b) Compute $Q((0,2), \text{North})$ using the formula $Q(s,a) = R + \gamma V(s')$.
    (c) Compute all four $Q((0,2), a)$ values, then verify $V(0,2) = \frac{1}{4}\sum_a Q((0,2), a) = 0.56$.

12. **Recycling Robot**: A robot is in state "high" ($s$).
    - Action "search": $p(\text{high, } 3 \mid \text{high, search}) = 0.6$, $p(\text{low, } 3 \mid \text{high, search}) = 0.4$.
    - Current values: $V(\text{high})=10, V(\text{low})=5$.
    Calculate $q(\text{high, search})$ with $\gamma = 0.8$.

13. **Gambler's Problem**: If a gambler has capital $s=50$, bets all of it ($a=50$), and $p_{\text{win}}=0.4$, calculate the expected value $V(50)$ given $V(0)=0$ and $V(100)=1$ with $\gamma=1$.

14. **Small MDP Cycle**: State $A \to B$ with $R=1$, $B \to A$ with $R=1$. Calculate $V(A)$ and $V(B)$ for a policy that always moves, with $\gamma=0.5$.

15. **Bellman Optimality**: In a state $S$, two actions $a_1, a_2$ exist.
    - $a_1$: $50\%$ chance of $R=10, V(S')=0$; $50\%$ chance of $R=0, V(S')=10$.
    - $a_2$: $100\%$ chance of $R=4, V(S')=2$.
    With $\gamma=1$, find $V^*(S)$.

16. **Effective Horizon**: If $\gamma = 0.9$, at what step $k$ does the reward $R_{t+k+1}$ contribute less than $10\%$ of its original value to the return $G_t$?

17. **Probability Derivation**: $p(s', r \mid s, a)$ is given for $r \in \{0, 10\}$. If $\sum_{s'} p(s', 10 \mid s, a) = 0.2$ and $\sum_{s'} p(s', 0 \mid s, a) = 0.8$, calculate $r(s, a)$.

18. **Policy Improvement**: Policy $\pi$ gives $V_\pi(s) = 10$. A new action $a'$ in state $s$ has $q_\pi(s, a') = 12$. If we change $\pi(s)$ to $a'$, what is the lower bound for the new value $V_{\pi'}(s)$ according to the Policy Improvement Theorem?

19. **Two-State Evaluation**: $S=\{1, 2\}$. $\pi(1)=a, \pi(2)=b$.
    - $1 \xrightarrow{a} 2$ with $R=10$.
    - $2 \xrightarrow{b} 1$ with $R=0$.
    $\gamma = 0.9$. Solve for $V(1)$ and $V(2)$.

20. **Max Stake**: In Gambler's problem with $s=60$ and goal $100$, what is the maximum possible stake $a$ if the gambler cannot bet more than their current capital and cannot bet more than what is needed to reach the goal?

21. In ε-greedy action selection, for the case of two actions and ε = 0.5, what is the probability that the greedy action is selected?

22. **Policy Iteration vs Value Iteration**: Consider a 3-state MDP with states $\{A, B, \text{Terminal}\}$ and two actions (left, right) per state. The dynamics are:
    - From $A$: **left** → stay at $A$, $R = -1$; **right** → go to $B$, $R = 0$
    - From $B$: **left** → go to $A$, $R = 0$; **right** → go to Terminal, $R = +5$
    - $\gamma = 0.9$

    **(a) Policy Iteration**: Starting from the initial policy $\pi(A) = \text{left}, \pi(B) = \text{left}$:
    - Perform **Policy Evaluation** (solve the Bellman equations exactly) to find $V^\pi(A)$ and $V^\pi(B)$.
    - Perform **Policy Improvement** by computing $Q(s,a)$ for all state-action pairs and selecting $\arg\max_a Q(s,a)$.
    - Repeat until the policy stabilizes. How many iterations does it take?

    **(b) Value Iteration**: Starting from $V(A) = 0, V(B) = 0$:
    - Perform sweeps using $V(s) \leftarrow \max_a [R(s,a) + \gamma V(s')]$.
    - Show each sweep until convergence (threshold $\theta = 0.01$). How many sweeps does it take?

    **(c)** Compare the two methods: which finds the optimal policy faster? Which requires solving a system of equations?

    **(d) Scaling up**: Now extend to a 5-state chain $\{A, B, C, D, E, \text{Terminal}\}$ with the same action structure (left = stay, R=-1; right = move forward, R=0 except E→Terminal gives R=+10). Using **iterative** policy evaluation ($\theta = 0.01$) instead of exact, how many total sweeps does Policy Iteration need vs Value Iteration? Why is the difference so dramatic?

23. **Policy Evaluation Convergence**: Using the same MDP as Q22, start with $V(A) = 0, V(B) = 0$ and perform iterative policy evaluation for the policy $\pi(A) = \text{right}, \pi(B) = \text{right}$ (the optimal policy). Show the values after each sweep until convergence ($\theta = 0.01$). How does this compare to directly solving the Bellman equations?

24. **Stochastic Gridworld — Value Iteration with Noisy Movement**: Consider the classic 3×4 gridworld (Russell & Norvig style):

    **Grid Layout:**
    ```
              Col 0    Col 1    Col 2    Col 3
            +--------+--------+--------+--------+
    Row 0   |  START |        |        | +1(G)  |
            +--------+--------+--------+--------+
    Row 1   |        |  WALL  |        | -1(P)  |
            +--------+--------+--------+--------+
    Row 2   |        |        |        |        |
            +--------+--------+--------+--------+
    ```

    **Rules:**
    - **Terminal states:** $(0,3)$ with reward $+1$ (Goal), $(1,3)$ with reward $-1$ (Pit). Episode ends upon entering these.
    - **Wall:** $(1,1)$ is impassable.
    - **Living reward:** $R = -0.04$ for every transition into a non-terminal state (battery drain).
    - **Stochastic movement:** Actions are noisy:
      - 80% probability: move in the intended direction.
      - 10% probability: move perpendicular (left of intended).
      - 10% probability: move perpendicular (right of intended).
      - If movement would hit a wall or boundary, the agent stays in place.
    - $\gamma = 0.9$, convergence threshold $\theta = 0.001$.

    **(a)** Illustrate how stochastic transitions work for state $(0,2)$ taking action East. What are the possible outcomes and their probabilities?

    **(b)** Perform Value Iteration. Show the grid of $V(s)$ values after each iteration (0 through convergence). How many iterations does it take?

    **(c)** Show the detailed Q-value computation for state $(0,2)$ at iteration 2, demonstrating how we pick $\max_a Q(s,a)$.

    **(d)** Draw the optimal policy (arrows showing best action in each state) derived from the converged values.

25. **Discount Factor and Optimal Policy**: Consider a 1D gridworld with 5 states in a line:

    ```
    [a] --- [b] --- [c] --- [d] --- [e]
    ```

    **Rules:**
    - **Actions:** East (move right), West (move left), and Exit (only available in terminal states $a$ and $e$).
    - **Terminal states:** $a$ (leftmost) and $e$ (rightmost). The agent can only collect reward by taking the Exit action in these states.
    - **Rewards:** $R(a, \text{Exit}) = 10$, $R(e, \text{Exit}) = 1$. All movement actions give $R = 0$.
    - **Transitions:** Deterministic.

    **(a)** For $\gamma = 1$, what is the optimal policy for each state?

    **(b)** For $\gamma = 0.1$, what is the optimal policy for each state?

    **(c)** For which value of $\gamma$ are actions West and East equally good when in state $d$?

26. **Race Car MDP — Policy Iteration vs Value Iteration**: A race car's engine has three temperature states:

    ```
    ┌──────────┐     ┌──────────┐     ┌──────────────┐
    │   Cool   │────▶│   Warm   │────▶│  Overheated  │
    │   (C)    │◀────│   (W)    │     │  (Terminal)   │
    └──────────┘     └──────────┘     └──────────────┘
    ```

    **States:** $S = \{\text{Cool}, \text{Warm}, \text{Overheated}\}$. Overheated is terminal ($V(\text{OH}) = 0$).

    **Actions:** Slow ($S$) and Fast ($F$).

    **Transition Probabilities $P(s' \mid s, a)$:**

    | State | Action | $P(\text{Cool})$ | $P(\text{Warm})$ | $P(\text{OH})$ |
    |-------|--------|:-:|:-:|:-:|
    | Cool | Slow | 0.5 | 0.5 | 0 |
    | Cool | Fast | 0.5 | 0.5 | 0 |
    | Warm | Slow | 0.5 | 0.5 | 0 |
    | Warm | Fast | 0 | 0 | 1.0 |

    **Rewards $R(s, a, s')$:**
    - Transition to a non-terminal state with action Slow: $R = +1$
    - Transition to a non-terminal state with action Fast: $R = +2$
    - Transition to Overheated: $R = -10$

    **Parameters:** $\gamma = 0.9$

    **(a) Policy Iteration:** Starting from the initial policy $\pi(\text{Cool}) = \text{Slow}, \pi(\text{Warm}) = \text{Slow}$:
    - Perform exact Policy Evaluation (solve Bellman equations) at each step.
    - Perform Policy Improvement by computing all Q-values.
    - Show each iteration until convergence.

    **(b) Value Iteration:** Starting from $V(\text{Cool}) = 0, V(\text{Warm}) = 0$:
    - Perform sweeps using $V(s) \leftarrow \max_a Q(s, a)$.
    - Show Q-values and V-values for each iteration until convergence ($\theta = 0.01$).

    **(c)** Compare the two methods: how many iterations does each take? Why is there such a dramatic difference?
