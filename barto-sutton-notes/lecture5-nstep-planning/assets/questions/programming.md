---
layout: post
---

# Programming Questions - Lecture 5 (n-step Bootstrapping & Planning)

1. **n-step TD Prediction:**
   Implement a Python function `n_step_td_prediction(V, episodes, n, alpha, gamma)` that performs n-step TD prediction for a simple random walk environment.
   - `V` is the initial value function (numpy array).
   - `episodes` is the number of episodes to run.
   - `n` is the number of steps.
   - `alpha` is the step size.
   - `gamma` is the discount factor.
   
2. **Dyna-Q Planning Loop:**
   Implement the planning part of the Dyna-Q algorithm. Given a `model` dictionary, a value function `Q`, and a set of `observed_states`, implement a function `dyna_q_planning(Q, model, n_planning, alpha, gamma)`.
   - The model should store `(s, a) -> (s_next, reward)`.
   - In each planning step, sample a random state-action pair from the model and perform a Q-learning update.

3. **Prioritized Sweeping Queue Update:**
   Implement the logic to update a priority queue in Prioritized Sweeping. Given a transition $(s, a, s', r)$, the magnitude of the TD error $\delta$, and a threshold $\theta$:
   - If $\mid \delta \mid > \theta$, add $(s, a)$ to the priority queue with priority $\mid \delta \mid$.
   - For all predecessors $(\bar{s}, \bar{a})$ of $s$, calculate the potential change in their values and add them to the queue if it exceeds $\theta$.
   
---
*Note: Solutions for these can be found in the `solutions/` folder.*
