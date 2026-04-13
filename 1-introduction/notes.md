# A journey into the world of RL 
## An introduction of the context
1. solves **sequential decision** making problems.
2. there is always an **objective**. reach an objective involves taking many actions in sequence. each action changing the environment.
3. example of balancing a flag pole. `slm-lab run --render`
4. environment : 
   - states
   - rewards for each state
   - model dynamics / transition probability (conditional probability distribution of state transition and rewards given an action is taken in a given state - $P(s',r|s,a)$
   - obstruction : states with negative / low rewards
   - objective 
5. agent : 
   - policy : function that maps state to action : for stochastic cases it would be conditional probability of actions, given a state - $\pi(a|s)$
   - action : the current state is changed, and a reward is received.
6. agent and environment are mutually exclusive
7. $(s_t, a_t, r_t)$ control loop : experience.
8. the control loop can repeat forever theoretically. However, they terminate after reaching a terminal state or a max number of steps t = T. time horizon from t=0 to t=T is known as an episode.
9.  a trajectory is a sequence of experiences in an episode : $\tau = (s_0,a_0,r_0), (s_1,a_1,r_1), (s_2,a_2,r_2), . . ., (s_T, a_T, r_t)$
10. an agent typically needs several episodes to come up with a good policy. agent's objective is to fine tune its policy to get the maximum return.

## Value functions 
1. **return - a retrospective approach - one episode at a time** : the discounted cumulative reward from time step $t$ to the **end of the episode**. $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$ where $\gamma \in [0,1]$ is the discount factor. $\gamma$ controls how much the agent values future rewards vs immediate rewards. $\gamma = 0$ makes the agent greedy (only cares about immediate reward), $\gamma = 1$ makes the agent far-sighted (values all future rewards equally).
$G_t$ is a random variable — even for a single trajectory, the rewards depend on actions sampled from $\pi(a|s)$ and transitions sampled from $P(s',r|s,a)$. The computed value of $G_t$ is a single realization of this random variable. Its expected value, expanding the probabilities explicitly: $$E_\pi[G_t | S_t = s] = \sum_{a} \pi(a|s) \sum_{s',r} P(s',r|s,a) \left[r + \gamma \, E_\pi[G_{t+1} | S_{t+1} = s']\right]$$ This is exactly $V^\pi(s)$ — the expected return is the state value function.
Recursive form: $$G_t = R_{t+1} + \gamma \, G_{t+1}$$ with $G_T = 0$. The return at time $t$ equals the immediate reward plus the discounted return from the next step.
2.  **state value function** $V^\pi(s)$ : the expected return starting from state $s$ and following policy $\pi$. $$V^\pi(s) = E_\pi [G_t | S_t = s] = E_\pi \left[\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \mid S_t = s\right]$$
**expanding the expectation** — the agent picks action $a$ with probability $\pi(a|s)$, then the environment transitions to state $s'$ with reward $r$ with probability $P(s',r|s,a)$. The future return from $s'$ is still an expectation: $$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s',r} P(s',r|s,a) \left[r + \gamma \, E_\pi[G_{t+1} | S_{t+1} = s']\right]$$
**recursive form (Bellman equation for $V^\pi$)** : $$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s',r} P(s',r|s,a) \left[r + \gamma \, V^\pi(s')\right]$$. The value of a state equals the expected immediate reward plus the discounted value of the next state, averaged over all actions (weighted by policy) and all possible transitions.
3.   **action Value function** $Q^\pi(s,a)$ : the expected return starting from state $s$, taking action $a$, and then following policy $\pi$. $$Q^\pi(s,a) = E_\pi [G_t | S_t = s, A_t = a] = E_\pi \left[\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$$
**expanding the expectation** — the action $a$ is already given, so we only sum over the environment dynamics. The environment transitions to state $s'$ with reward $r$ with probability $P(s',r|s,a)$. The future return from $s'$ is still an expectation: $$Q^\pi(s,a) = \sum_{s',r} P(s',r|s,a) \left[r + \gamma \, E_\pi[G_{t+1} | S_{t+1} = s']\right]$$
**recursive form (Bellman equation for $Q^\pi$)** : $$Q^\pi(s,a) = \sum_{s',r} P(s',r|s,a) \left[r + \gamma \sum_{a'} \pi(a'|s') \, Q^\pi(s',a')\right]$$. The value of taking action $a$ in state $s$ equals the expected immediate reward plus the discounted action value at the next state, averaged over the next action chosen by the policy.
4.  **relationship**: $V^\pi(s) = \sum_{a} \pi(a|s) \, Q^\pi(s,a)$ — the state value is the expected action value over all actions weighted by the policy.
5.  **return before vs after an episode** : before the episode plays out, the return from state $s$ is unknown — it depends on what actions the policy will take and how the environment will respond. The best we can say is the *expected* return, which is $$V^\pi(s) = E_\pi[G_t | S_t = s]$$. **This is a prediction.** After the episode completes, we have the actual rewards $R_{t+1}, R_{t+2}, \dots, R_T$ and can compute the **realized return**: $$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$. **This is a single sample of the random variable $G_t$ after the episode has completed**.
6.  **timeline of what is available when** :
   - **during an episode** : rewards $R_{t+1}$ are observed step by step, but $G_t$ is not yet available — we don't know the future rewards. The only thing we can work with is a prediction: $V^\pi(s)$ or $Q^\pi(s,a)$, but these require either a learned value function (from prior episodes) or known dynamics $P(s',r|s,a)$ to solve the Bellman equations.
   - **after the episode** : all rewards are known. We compute $G_t$ backwards from the terminal state using $G_t = R_{t+1} + \gamma \, G_{t+1}$ with $G_T = 0$. This gives a single realized sample of return for every time step in the episode.
   - **after many episodes** : we accumulate many samples of $G_t$ for each state. Averaging these samples gives an estimate of $V^\pi(s)$ — this is Monte Carlo.
7.  **why one episode is not enough for $V^\pi(s)$ and $Q^\pi(s,a)$** : even with a fixed policy, both $\pi(a|s)$ and $P(s',r|s,a)$ are stochastic. From the same state $s$, different episodes produce different trajectories with different rewards, giving different $G_t$ values. One episode gives one sample — one roll of the dice. The true $V^\pi(s)$ is the average over *all possible trajectories* from $s$, not the return from any single one. One episode would only be sufficient if both the policy and the environment were fully deterministic — then every trajectory from state $s$ would be identical, and a single $G_t$ would equal $V^\pi(s)$ exactly.
8.  **monte carlo estimation** : if we run many episodes and collect the realized return $G_t$ every time we visit state $s$, the average of those realized returns converges to the true state value: $$V^\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$ where $N(s)$ is the number of times state $s$ was visited and $G_t^{(i)}$ is the realized return from the $i$-th visit. This is the **Monte Carlo method** — it estimates the expected value by averaging samples. It requires no knowledge of $P(s',r|s,a)$ (model-free), only completed episodes. The same idea applies to $Q^\pi(s,a)$: collect realized returns every time action $a$ is taken in state $s$, and average them: $$Q^\pi(s,a) \approx \frac{1}{N(s,a)} \sum_{i=1}^{N(s,a)} G_t^{(i)}$$
9. **[environment in work](./env.ipynb)**
10. **[mdp definition](./mdp-definition.md)**
11.  ![](/assets/mdp.svg)

## From random actions to optimal policy — the dynamic programming transition

### The random agent baseline ([env.ipynb](./env.ipynb))

In `env.ipynb`, the agent picks actions uniformly at random: `action = env.action_space.sample()`. There is no learning, no memory, no strategy. The policy is:
$$\pi(a|s) = \frac{1}{|\mathcal{A}|} \quad \forall \, s, a$$

This is useful as a baseline to observe what the environment looks like — what states are visited, what rewards come back, how episodes end. But the returns are poor. In Taxi-v3, for example, the random agent almost never delivers the passenger before the 200-step truncation. Every episode accumulates step penalties (-1 per step) and illegal-action penalties (-10), ending with deeply negative returns. The agent has no way to improve because it never uses the information it collects.

**What is missing:** the agent needs a mechanism to evaluate how good its current behavior is, and then change its behavior to be better.

### The dynamic programming agent ([dynamic-prog/](./dynamic-prog/dynamic_programming_case_study.ipynb))

Dynamic programming solves this by exploiting the **full environment model** $P(s',r|s,a)$. Unlike the random agent that must *run* episodes to see what happens, DP can *compute* what would happen under any policy without taking a single step.

The algorithm — **policy iteration** — works in a loop:

**Step 1 — Policy Evaluation: how good is the current policy?**
Start with the same uniform random policy as `env.ipynb`. Compute $V^\pi(s)$ for every state by solving the Bellman equation:
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a)\left[r + \gamma V^\pi(s')\right]$$
This tells us the expected return from every state *if we keep following the current policy*. For the initial uniform policy on the 3x3 grid, all V values are negative — the random policy is bad.

**Step 2 — Q from V: compare individual actions**
$V^\pi(s)$ tells us how good a *state* is, but to improve the policy we need to compare *actions*. Convert V to Q:
$$Q^\pi(s,a) = \sum_{s',r} P(s',r|s,a)\left[r + \gamma V^\pi(s')\right]$$
Now for each state, we can see which action leads to the highest expected return. This is the critical bridge — Q breaks down the state value into per-action values so we can identify which actions are better than others.

**Step 3 — Policy Improvement: shift probability to the best action**
For each state, find $\argmax_a Q^\pi(s,a)$ — the action(s) with the highest Q value. Build a new policy that puts most probability mass on those actions:
- Best action(s) get probability $\frac{1-\epsilon}{|\text{best actions}|} + \frac{\epsilon}{|\mathcal{A}|}$
- Other actions get just $\frac{\epsilon}{|\mathcal{A}|}$ (exploration floor)

This is no longer uniform — the policy now *prefers* actions that lead to higher returns.

**Step 4 — Repeat** until the greedy actions stop changing (the policy is stable).

### How the strategy improves — the 3x3 grid example

The grid has state 0 (top-left) as start and state 8 (bottom-right, G) as goal. Step reward is -1, goal reward is +10.

**Iteration 0 — evaluating the uniform policy:**
| | Col 0 | Col 1 | Col 2 |
|---|---|---|---|
| Row 0 | -5.92 | -5.02 | -3.77 |
| Row 1 | -5.02 | -3.14 | 0.25 |
| Row 2 | -3.77 | 0.25 | G |

All values are negative or near zero. The random policy wanders aimlessly, accumulating -1 penalties. Q values are computed, and for state 0, Q shows that RIGHT and DOWN (both -5.60) are better than UP and LEFT (both -6.25). This makes sense — RIGHT and DOWN move toward the goal. The policy is updated: RIGHT and DOWN get high probability in state 0.

**Iteration 1 — evaluating the improved policy:**
| | Col 0 | Col 1 | Col 2 |
|---|---|---|---|
| Row 0 | 2.71 | 4.35 | 6.22 |
| Row 1 | 4.35 | 6.46 | 8.93 |
| Row 2 | 6.22 | 8.93 | G |

All values are now **positive**. The improved policy reaches the goal reliably enough that the +10 goal reward outweighs the step penalties. The greedy actions haven't changed from iteration 0 — the policy is already stable. Convergence in just 2 iterations.

The final policy points every state toward the goal:
```
+---+---+---+
|>v | v | v |
+---+---+---+
| > |>v | v |
+---+---+---+
| > | > | G |
+---+---+---+
```

### The key transition: why DP works where the random agent cannot

| | Random agent (env.ipynb) | DP agent (dynamic-prog/) |
|---|---|---|
| **Policy** | Fixed uniform $\pi(a\|s) = 1/\|\mathcal{A}\|$ | Iteratively improved toward optimal |
| **Uses model $P(s',r\|s,a)$?** | No — just samples actions and observes | Yes — computes exact expectations |
| **Learning** | None — same policy forever | Policy evaluation + improvement loop |
| **Requires episodes?** | Yes — must run the environment | No — plans entirely from the model |
| **Limitation** | No improvement possible | Requires knowing the full model |

The random agent in `env.ipynb` gives us the **floor** — what happens with zero intelligence. Dynamic programming gives us the **ceiling** for this MDP — the optimal policy computed exactly from the known model. The gap between them is what learning algorithms (Monte Carlo, TD, Q-learning) try to close *without* knowing $P(s',r|s,a)$.