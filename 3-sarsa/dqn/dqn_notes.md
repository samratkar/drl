### DQN 

$$Target : r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

$$Loss = \left( \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{Where we want to be (Target)}} - \underbrace{Q(s, a)}_{\text{Where we are (Prediction)}} \right)^2$$

### Q Learning - Traditional 

$$Q(s, a) \leftarrow \underbrace{Q(s, a)}{\text{Old Value}} + \alpha \underbrace{\left[ \overbrace{r + \gamma \max{a'} Q(s', a')}^{\text{Target (The Truth)}} - \overbrace{Q(s, a)}^{\text{Prediction}}
  \right]}_{\text{TD Error}}$$

How it converges from Zero (Step-by-Step)
Imagine a simple grid where reaching the goal gives a reward of +100. All $Q(s, a) = 0$ initially.

1. Step 1 (The First Reward): The agent wanders randomly until it accidentally hits the goal.
    * $r = 100$, $s'$ is the "terminal" state (where $Q(s', a') = 0$ by definition).
    * $Q(s, a) \leftarrow 0 + \alpha [100 + \gamma(0) - 0] = \mathbf{10}$.
    * Result: The state right before the goal now has a Q-value of 10.

2. Step 2 (The Back-propagation): On the next episode, the agent reaches the state before the one that now has 10.
    * $r = 0$, but $\max Q(s', a') = 10$.
    * $Q(s_{prev}, a) \leftarrow 0 + \alpha [0 + \gamma(10) - 0] = \mathbf{0.9}$ (assuming $\gamma=0.9, \alpha=0.1$).
    * Result: The "knowledge" of the reward starts flowing backward through the table, one step at a time.

3. Step 3 (Convergence): After visiting these states thousands of times, the numbers stop changing because the TD Error becomes Zero. When Target - Prediction = 0, the update stops, and you have reached the
    "optimal" Q-table.
