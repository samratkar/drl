---
layout: post
---

# Temporal Difference (TD) Learning - Detailed Solutions

---

## Chapter 1: Foundational TD Prediction (TD(0))

![TD Backup Diagram](../diagrams/td_backup_diagram.svg)

### Problem 1: The Commuter's Dilemma (MC vs. TD)

**Scenario:**
A daily commuter estimates their travel time from home to the office.
- **Agent:** The Commuter.
- **Environment:** The city traffic system.
- **State $s_0$:** Home. Initial estimate $V(s_0)=45$ min.
- **State $s_1$:** Highway entrance. New estimate for remainder = 20 min.
- **Actual outcome:** Time spent $s_0 \rightarrow s_1$ is 15 min.

**Tasks:**
a) Identify the states and rewards in this backup.
b) Formulate the TD error ($\delta_t$) for the transition $s_0 \rightarrow s_1$.
c) Explain why the commuter can update their estimate $V(s_0)$ before reaching the office.

**Solution:**

**a) Identification:**
- **States:** $s_0$ (Home), $s_1$ (Highway Entrance).
- **Reward ($r_{t+1}$):** The cost of the transition, which is $-15$ minutes (time elapsed is a negative reward in a minimization/cost setting).

**b) TD Error Calculation:**
The TD error formula is: $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$.
Assuming $\gamma = 1$ (undiscounted time):
- $r_{t+1} = -15$
- $V(s_1) = -20$ (The estimate for the remaining journey is a negative value in a cost-to-goal setting).
- $V(s_0) = -45$
$$\delta_t = (-15) + (-20) - (-45)$$
$$\delta_t = -35 + 45 = +10$$
*Interpretation:* The agent was pleasantly surprised. The trip is actually taking 10 minutes less than expected based on the $s_1$ observation.

**c) Conceptual Difference:**
In **Monte Carlo (MC)** methods, the commuter would have to wait until they actually parked at the office (reached the terminal state) to know the actual return $G_t$ and update $V(s_0)$. In **TD Learning**, the agent **bootstraps**: it uses the estimate of the next state ($V(s_1)$) and the immediate reward to update the previous state's value immediately. This allows for online learning during the episode.

---

### Problem 2: Smart Thermostat (Value Prediction)

**Question:**
A smart thermostat uses TD(0) to predict the energy cost ($V$) required to reach a target temperature.
- **Transition:** $20^\circ C$ ($s$) to $22^\circ C$ ($s'$).
- **Estimates:** $V(20^\circ C) = 50$, $V(22^\circ C) = 30$.
- **Reward ($r$):** $-5$.
- **Params:** $\alpha = 0.2$, $\gamma = 0.9$.

**Solution:**

**Formula:** $V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]$

1.  **Calculate the TD Target:**
    $$\text{Target} = r + \gamma V(s')$$
    $$\text{Target} = -5 + (0.9 \times 30) = -5 + 27 = 22$$

2.  **Calculate the Update:**
    $$V(20^\circ C) \leftarrow 50 + 0.2 \times [22 - 50]$$
    $$V(20^\circ C) \leftarrow 50 + 0.2 \times [-28]$$
    $$V(20^\circ C) \leftarrow 50 - 5.6 = 44.4$$

**Result:** The new predicted cost from $20^\circ C$ is **44.4 units**.

---

### Problem 3: The "Wait" Reward (TD Error)

**Question:**
In a stock trading scenario, an agent holds a stock.
- $V(s_t) = 10$, $V(s_{t+1}) = 12$, $r_{t+1} = 0$, $\gamma = 1.0$.
- Calculate $\delta_t$. Is the surprise positive or negative?

**Solution:**

**TD Error Formula:** $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$
$$\delta_t = 0 + 1.0(12) - 10 = +2$$

**Interpretation:**
The agent feels a **positive surprise** (positive TD error). Even though no immediate reward was received ($r=0$), the "news" from the next state is better than expected (the future value increased by 2). This demonstrates that TD learning updates based on changes in future expectations, not just immediate rewards.

---

### Problem 4: Autonomous Braking (Safety Margin)

**Question:**
- $V(10m) = 0$, $V(5m) = -200$.
- $r = -100$, $\alpha = 0.5$, $\gamma = 0.8$.
- Calculate new $V(10m)$.

**Solution:**

1.  **Identify Components:**
    - Current State value: $V(s) = 0$.
    - Next State value: $V(s') = -200$.
    - Reward: $r = -100$.

2.  **Apply TD(0) Update:**
    $$V(10m) \leftarrow 0 + 0.5 \times [-100 + 0.8(-200) - 0]$$
    $$V(10m) \leftarrow 0 + 0.5 \times [-100 - 160]$$
    $$V(10m) \leftarrow 0 + 0.5 \times [-260] = -130$$

**Insight:**
By updating $V(10m)$ to **-130**, the agent now realizes that being at "10 meters" is likely to lead to a dangerous "-200" state. Even if the current state felt "safe" ($V=0$), the TD update propagates the danger backwards.

---

### Problem 5: Delivery Drone (Path Prediction)

**Question:**
- Flies $s_1 \rightarrow s_2$, $r = -1$.
- $V(s_1) = 10$, $V(s_2) = 15$. $\alpha = 1$, $\gamma = 1$.
- a) New $V(s_1)$? b) Final convergence if $s_2 \rightarrow \text{Goal} (+100)$?

**Solution:**

**a) Immediate Update:**
$$V(s_1) \leftarrow 10 + 1.0 \times [-1 + 1.0(15) - 10]$$
$$V(s_1) \leftarrow 10 + [14 - 10] = 14$$

**b) Convergence Analysis:**
In TD prediction, $V(s)$ converges to the expected return.
If $s_2$ eventually leads to a terminal goal of $+100$ with a step cost of $-1$, then:
- $V(s_2) \rightarrow 100$ (since it's the step before goal).
- $V(s_1) \rightarrow r + V(s_2) = -1 + 100 = 99$.
Thus, $V(s_1)$ will eventually converge to **99**.

---

## Chapter 2: TD Control (SARSA, Q-Learning, Expected SARSA)

![SARSA and Q-Learning Backups](../diagrams/sarsa_q_backup.svg)

### Problem 6: The Cliff Walking Safety (Concept)

**Scenario:**
- Falling off cliff: $r = -100$, reset.
- Step cost: $r = -1$.
- SARSA vs. Q-Learning.

**Solution:**

**a) Classification:**
- **SARSA:** **On-policy** control. It updates $Q(s, a)$ based on the *actual* next action $a'$ taken by the current policy (including exploratory steps).
- **Q-Learning:** **Off-policy** control. It updates $Q(s, a)$ based on the *best possible* next action $\max_{a'} Q(s', a')$, regardless of what action the current policy actually selects.

**b) Safety during Training:**
SARSA is safer during training because it takes into account the agent's exploration strategy ($\epsilon$-greedy). Since the agent occasionally takes random actions, it might accidentally "fall off" the cliff while exploring. SARSA "sees" these accidental falls in its updates and learns to stay far enough away from the edge so that a random exploratory step doesn't lead to a $-100$ reward.

**c) Optimality:**
- **Q-Learning** learns the **true optimal policy** ($\pi^*$) that follows the very edge of the cliff.
- **SARSA** learns a policy that is **optimal given its exploration** (a "safe" policy). If $\epsilon$ were reduced to 0, SARSA would eventually converge to the same optimal path as Q-learning.

---

### Problem 7: E-commerce Discounts (Q-Learning)

**Question:**
- $s = \text{Med}$, $a = \text{10\% discount}$, $s' = \text{High}$, $r = +50$.
- $Q(High, 0\%) = 100$, $Q(High, 10\%) = 80$, $Q(High, 20\%) = 60$.
- $\alpha = 0.1$, $\gamma = 0.9$, current $Q(Med, 10\%) = 40$.

**Solution:**

**Q-Learning Formula:** $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$

1.  **Find the maximum Q-value in the next state:**
    $$\max_{a'} Q(High, a') = \max(100, 80, 60) = 100$$

2.  **Calculate the TD Target:**
    $$\text{Target} = r + \gamma \max_{a'} Q(s', a')$$
    $$\text{Target} = 50 + 0.9(100) = 50 + 90 = 140$$

3.  **Apply the Update:**
    $$Q(Med, 10\%) \leftarrow 40 + 0.1 \times [140 - 40]$$
    $$Q(Med, 10\%) \leftarrow 40 + 0.1 \times [100] = 40 + 10 = 50$$

**Result:** The updated value is **50**.

---

### Problem 8: Robotic Warehouse (SARSA Update)

**Question:**
- $s \rightarrow s'$, $r = -2$, $a' = \text{Identify}$.
- $Q(s, a) = 10$, $Q(s', \text{Identify}) = 20$, $Q(s', \text{Shelf 6}) = 25$ (Greedy).
- $\alpha = 0.5$, $\gamma = 0.8$.

**Solution:**

**SARSA Formula:** $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$

1.  **Identify the specific $a'$ chosen:**
    The robot chose "Identify Item", so we use $Q(s', \text{Identify}) = 20$.
    *(Note: We ignore the greedy action value 25 because SARSA is on-policy).*

2.  **Apply the Update:**
    $$Q(s, a) \leftarrow 10 + 0.5 \times [-2 + 0.8(20) - 10]$$
    $$Q(s, a) \leftarrow 10 + 0.5 \times [-2 + 16 - 10]$$
    $$Q(s, a) \leftarrow 10 + 0.5 \times [4] = 10 + 2 = 12$$

**Result:** The updated value is **12**.

---

### Problem 9: Stochastic Wind (Expected SARSA)

**Question:**
- $Q(s, a) = 30$, $r = 10$, $\alpha = 0.2$, $\gamma = 1.0$.
- In $s'$, $\pi$ is $\{a_1: 0.8, a_2: 0.2\}$ with $Q$ values $\{50, 20\}$.

**Solution:**

**Expected SARSA Formula:** $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a') - Q(s, a)]$

1.  **Calculate the Expected Value at $s'$:**
    $$E[Q(s', \cdot)] = (0.8 \times 50) + (0.2 \times 20)$$
    $$E[Q(s', \cdot)] = 40 + 4 = 44$$

2.  **Calculate the TD Target:**
    $$\text{Target} = 10 + 1.0(44) = 54$$

3.  **Apply the Update:**
    $$Q(s, a) \leftarrow 30 + 0.2 \times [54 - 30]$$
    $$Q(s, a) \leftarrow 30 + 0.2 \times [24] = 30 + 4.8 = 34.8$$

**Comparison:** Standard SARSA would have updated using either 50 or 20 (based on a random sample). Expected SARSA reduces variance by using the average over all possible next actions.

---

### Problem 10: Maximization Bias (The Double-Door Mystery)

**Solution:**

**a) Reasoning:**
The agent picks the action with the maximum estimated value: $\max(Q(s, A), Q(s, B))$. Because the rewards are noisy, some trials will randomly produce high positive values and others low negative values. By always taking the *maximum* of these noisy estimates, the agent is biased towards actions that happened to get "lucky" noise, even if the true mean is 0.

**b) Phenomenon:**
This is called **Maximization Bias**.

**c) Double Q-Learning:**
Double Q-learning maintains two independent estimates $Q_1$ and $Q_2$ for each state-action pair. 
- One estimate (say $Q_1$) is used to determine the greedy action: $a^* = \text{argmax}_a Q_1(s, a)$.
- The other estimate ($Q_2$) is used to provide the value for that action: $Q_2(s, a^*)$.
Since $Q_1$ and $Q_2$ are independent, the "luck" (noise) in $Q_1$ that made $a^*$ look good is unlikely to be present in $Q_2$, thus eliminating the systematic overestimation.

---

## Chapter 3: Advanced TD Concepts

![n-Step TD and Eligibility Traces](../diagrams/advanced_td_diagrams.svg)

### Problem 11: Ride-Sharing ETA (n-Step TD)

**Scenario:**
- $t=0 \rightarrow t=3$: $r_1 = -2, r_2 = -3, r_3 = -1$.
- $V(t=0) = 15, V(t=3) = 8, \gamma = 1.0, \alpha = 0.5$.

**Solution:**

**a) 3-Step TD Target ($G_{t:t+3}$):**
The $n$-step return is the sum of the next $n$ rewards plus the discounted value of the state reached after $n$ steps.
$$G_{0:3} = r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 V(s_3)$$
$$G_{0:3} = (-2) + (1.0)(-3) + (1.0)^2(-1) + (1.0)^3(8)$$
$$G_{0:3} = -2 - 3 - 1 + 8 = +2$$

**b) General $n$-step Return Formula:**
$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})$$

**c) Immediate Update:**
$$V(s_0) \leftarrow V(s_0) + \alpha [G_{0:3} - V(s_0)]$$
$$V(s_0) \leftarrow 15 + 0.5 \times [2 - 15]$$
$$V(s_0) \leftarrow 15 + 0.5 \times [-13] = 15 - 6.5 = 8.5$$

---

### Problem 12: Packet Routing (Off-Policy Prediction)

**Scenario:**
- $r = -10, V(s_t) = -50, V(s_{t+1}) = -35, \gamma = 1.0$.
- $\pi(a \mid s) = 0.9, b(a \mid s) = 0.3$.

**Solution:**

1.  **Calculate Importance Sampling Ratio ($\rho_t$):**
    $$\rho_t = \frac{\pi(a_t \mid s_t)}{b(a_t \mid s_t)} = \frac{0.9}{0.3} = 3.0$$

2.  **Apply Off-Policy Update Rule:**
    $$V(s_t) \leftarrow V(s_t) + \alpha \rho_t [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$
    Assuming $\alpha = 0.1$:
    $$V(s_t) \leftarrow -50 + 0.1 \times (3.0) \times [-10 + 1.0(-35) - (-50)]$$
    $$V(s_t) \leftarrow -50 + 0.3 \times [-10 - 35 + 50]$$
    $$V(s_t) \leftarrow -50 + 0.3 \times [5] = -50 + 1.5 = -48.5$$

**Interpretation:** The ratio $\rho$ scales the update. Since the action taken is 3x more likely under the target policy than the behavior policy, the "news" from this transition is given 3x more weight.

---

### Problem 13: Credit Assignment (Eligibility Traces)

**Solution:**

**a) Eligibility Trace ($z_t(s)$):**
It is a memory variable associated with each state that tracks how recently and frequently a state was visited. It decays over time by $\gamma \lambda$ and increases by 1 when the state is visited.

**b) Value at $t=2$:**
If $s$ is visited for the first time: $z_2(s) = 1.0$.

**c) Value at $t=3$:**
If $s$ is NOT visited: $z_3(s) = \gamma \lambda z_2(s) = 0.9 \times 0.8 \times 1.0 = 0.72$.

**d) Bridging the Gap:**
- If $\lambda = 0$, TD($\lambda$) becomes **TD(0)** (only the immediate predecessor gets an update).
- If $\lambda = 1$, TD($\lambda$) becomes **Monte Carlo** (in the offline case), where the reward at the end of the episode is propagated all the way back to every state visited, scaled by their decay.
- Values in between allow for a flexible trade-off between bias (TD) and variance (MC).

---

### Problem 14: Batch TD (Offline Stability)

**Solution:**

**Stability:**
In Online TD, updates from a single noisy transition can temporarily pull the value function in the wrong direction. Batch TD processes all transitions in the dataset multiple times, effectively "smoothing" out the noise. It calculates the cumulative gradient before making any changes, preventing the estimator from oscillating.

**Convergence:**
Batch TD converges to the **Certainty-Equivalence Estimate**. This is the value function that would be optimal if the observed transitions and rewards were exactly correct (i.e., if the maximum likelihood model of the environment was perfectly accurate).

---

### Problem 15: The Deadly Triad

**Solution:**

**a) tabular Q-learning components:**
- **Bootstrapping:** YES (uses $\max Q(s', a')$).
- **Off-policy Training:** YES (updates greedy values while following $\epsilon$-greedy).
- **Function Approximation:** **NO** (tabular methods store each state separately). 
*Note: Tabular Q-learning is stable because it lacks function approximation.*

**b) Instability in Deep RL:**
When you combine all three:
1.  **Function Approximation** (Neural Networks) makes states "overlap" (updating one affects others).
2.  **Bootstrapping** means you are updating a guess based on another guess.
3.  **Off-policy** means the data distribution doesn't match the policy being learned.
This creates a positive feedback loop where errors in the network's output are used as targets for its own training, causing the $Q$-values to spiral out of control and diverge to infinity.

---

## Chapter 4: Special Cases & Final Synthesis

![The Reinforcement Learning Landscape](../diagrams/td_landscape.svg)

### Problem 16: TD vs. DP (Relationship)

**Solution:**

**Sampling vs. Expectation:**
- In **Dynamic Programming**, the update is an **expectation**. It requires a full model $P(s' \mid s, a)$ to calculate the average of all possible next states and rewards.
- In **Temporal Difference learning**, the update is a **sample**. Instead of calculating the average over all possibilities, it takes the actual reward $R$ and next state $s'$ experienced in a single trial.

**Why TD works without a model:**
Because TD uses samples, it doesn't need to know the transition probabilities $P$ beforehand. By taking enough samples over time, the average of these sample updates mathematically converges to the same result as the DP expectation update (according to the Law of Large Numbers). This makes TD "model-free."

---

### Problem 17: Learning Rate Impact (Alpha)

**Solution:**

**Condition A ($\alpha = 0.01$):**
- **Pros:** Handles random noise very well. The value function is stable and reflects long-term averages.
- **Cons:** Extremely slow to react to changes. If the market shifts suddenly, the agent will still be using "stale" information from weeks ago.

**Condition B ($\alpha = 0.8$):**
- **Pros:** Tracks sudden market shifts rapidly. The agent is highly responsive to new "news."
- **Cons:** Very sensitive to noise. A single "weird" day in the market will cause a massive, likely incorrect, swing in the value estimates.

**Recommendation:** For a highly volatile market, a moderate $\alpha$ (e.g., 0.1 to 0.3) is usually preferred, often starting higher and decaying as the agent gains confidence.

---

### Problem 18: Online vs. Offline Updates

**Solution:**

**Online TD Advantage:**
**Computational Efficiency:** You don't need to store a massive history of transitions. You process a transition, update $V$, and can immediately discard the data. This is crucial for real-time systems with limited memory (like embedded sensors).

**Offline TD Reason:**
**Data Efficiency / Stability:** Offline updates calculate the "average" direction of all transitions in the episode before moving the value function. This prevents the estimator from being "pulled" in different directions by early noisy steps in a long episode, leading to a more stable convergence on that specific dataset.

---

### Problem 19: The TD-MC Spectrum (Hybrid returns)

**Solution:**

**Option 1: 5-step TD:**
This looks ahead by exactly 5 rewards before bootstrapping from the estimate of the state reached at step 6. It reduces the bias of TD(0) by using more real data, while still maintaining the efficiency of bootstrapping.

**Option 2: TD($\lambda$):**
This uses a weighted average of *all* possible $n$-step returns ($1$-step, $2$-step, ..., up to full MC). By setting $\lambda = 0.5$, it gives more weight to recent steps and exponentially less weight to steps far in the future.

**Synthesis:** Both methods reduce "bootstrapping error" by incorporating more real experience into the update target, but they stop short of the high variance associated with full Monte Carlo.

---

### Problem 20: Self-Improving Chatbot (Comprehensive Case)

**Solution:**

**a) Reward Function ($r$):**
- $-1$ for every turn taken (incentivizes speed).
- $-10$ if "High Frustration" is detected in the next state.
- $+100$ upon reaching "Goal Reached" (Terminal).
*Goal:* Minimize total cost (turns + frustration).

**b) Algorithm Choice:**
**SARSA (On-policy)** is better here. 
*Reason:* Since the bot is interacting with real customers, "risky" or "creative" answers could lead to high frustration. SARSA learns the value of the policy it is *actually following* (including those risky exploratory steps). It will learn to be more "cautious" during training. Q-learning would ignore the cost of its own exploration, potentially leading to a bot that is too aggressive during the learning phase.

**c) State Space ($S$):**
A tuple representing:
- (Current Intent Category, Number of Turns, Frustration Level {Low, High}, Information Provided Flag).
*Example State:* ("Refund Request", Turn 2, "High", "Order ID Provided").

---
