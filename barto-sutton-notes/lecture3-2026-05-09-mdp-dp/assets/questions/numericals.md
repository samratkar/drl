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

11. **Gridworld Boundary Reward**: Moving north from $(0, 3)$ hits the boundary. The rule is: stay in $(0, 3)$ and receive $R=-1$. If $\gamma=0.9$ and $V(0,3)=5$, calculate the contribution of the "north" action to the Bellman equation for $V(0,3)$.

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
