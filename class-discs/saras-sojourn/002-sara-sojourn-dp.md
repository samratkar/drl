---
tags : [drl-class-notes]
title : "002 - Sara's sojourn - Dynamic Programming"
classes : ["2026-05-02"]
---

## The Ranger’s Map (Dynamic Programming)

Sara finds a "perfect map" (the Model $\mathcal{P}$). It tells her every probability $p(s', r | s, a)$. Because Sara has this perfect model, she can perform **Dynamic Programming (DP)**—an offline planning process. She doesn't need to take a single step to know how to act; she just needs to sit by her campfire and calculate.

### 1. The Bellman Equations: Expectation vs. Optimality

Sara distinguishes between two types of calculations:

*   **Bellman Expectation Equation:** This tells Sara the value of her *current* way of walking (her policy $\pi$). It asks: "If I keep following this specific plan, how much total reward will I get from state $s$?"
    $$V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_\pi(s') \right]$$
*   **Bellman Optimality Equation:** This tells Sara the *best possible* value she could ever get. It asks: "If I always make the best possible choice at every step, what is the maximum reward I can get from state $s$?"
    $$V^*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V^*(s') \right]$$

The key difference is the **$\max$** operator. The Expectation equation averages over actions (following $\pi$), while the Optimality equation selects the single best action.

### 2. Solving via Iteration

Since these equations are recursive (the value of $s$ depends on the value of $s'$), Sara solves them using **Iterative Methods**.

#### Policy Evaluation
To find $V_\pi$, Sara starts with an initial guess for all states (e.g., all zeros). She then repeatedly "sweeps" through every state on her map, updating her guess:
$$V_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_k(s') \right]$$
She repeats this until the values stop changing (convergence).

#### Value Iteration
To find the optimal policy directly, she turns the Bellman Optimality Equation into an update rule:
$$V_{k+1}(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_k(s') \right]$$

### 3. The Target and the "Learning Rate"

In every update, Sara looks at the **Bellman Target**:
$$\text{Target} = r + \gamma V(s')$$
This is what she *expects* the value to be based on the rewards and the estimated value of the next state.

**How is she "learning"?**
*   **The Learning Rate ($\alpha$):** In pure Dynamic Programming, Sara has the "truth" (the model). Therefore, she doesn't need to average her findings over many samples. She sets her **learning rate effectively to 1.0**. This means she completely replaces her old guess with the new calculation (the target).
*   **Bootstrapping:** The "learning" happens because Sara uses her guess of the future ($V(s')$) to update her current guess ($V(s)$). As she iterates, the true values of terminal states (like the forest exit) "ripple" backward through the map until every state knows its true worth.

### 4. Is it Recursion or a Circular Loop?

You might notice a problem: If Sara can walk in a circle (State A $\leftrightarrow$ State B), then the value of A depends on B, and the value of B depends on A. 

**"But wait," you say, "I have a terminal state! Can't I just backtrack from there?"**

Yes! That is exactly what DP is doing. However, "simple" backtracking (like a single reverse-walk) fails because of two things:

1.  **Probabilities (The Map is blurry):** From State B, Sara might have a 50% chance of reaching the exit (Terminal) and a 50% chance of being blown back to State A. Now, $V(B)$ depends on *both* the exit and State A. You can't solve for B until you know A, and you can't solve for A until you know B.
2.  **Cycles (The Infinite Loop):** In a Directed Acyclic Graph (a map with no loops), you could just start at the end and work backward in one pass. But RL environments often have loops. If Sara is stuck in a loop, simple recursion would follow that loop forever and never actually reach the terminal state to begin the backtracking!

**How Sara solves the circularity:**

*   **System of Equations:** Instead of a single "backtracking walk," Sara treats the map as a system of simultaneous equations. She doesn't try to find the "end" and walk back; she tries to find a set of values for *all* states that are self-consistent.
*   **The "Anchor" (Terminal States):** You are correct—the terminal state is the anchor. Even if $A$ and $B$ are in a circular loop, their values are ultimately tied to how close they are to that terminal zero.
*   **The "Safety Net" (Discount Factor $\gamma$):** If Sara stays in a circle forever, the reward gets multiplied by $\gamma$ every loop. Since $\gamma < 1$, the "infinite" value eventually shrinks to zero. This ensures the loop doesn't have an "infinite" price tag.

### 5. Why Iteration?

If you have 3 states, you can solve the equations with pen and paper. If you have 1,000,000 states, you use **Successive Approximation**. 

Sara starts with a guess (all zeros) and uses the Bellman Equation to "sweep" the map. 
*   In the first sweep, only the states right next to the exit "see" the truth. 
*   In the second sweep, the states two steps away "see" the truth through the first states. 
*   **This is "Backtracking" happening over time.** 

Each "sweep" (iteration) ripples the truth one step further away from the terminal anchor until the entire map reaches a **Fixed Point**—a state of mathematical harmony where no value needs to change anymore.

### 6. The Algorithm in Python

Here is how Sara’s campfire calculations look in code. Note how the "learning" is simply setting the value to the **Bellman Target**.

```python
import numpy as np

def policy_evaluation(states, actions, model, policy, gamma=0.9, theta=1e-6):
    """
    states: list of states [0, 1, 2, ...]
    model(s, a): returns list of (prob, next_state, reward)
    policy(s, a): returns probability of taking action 'a' in state 's'
    """
    # 1. Initialize Sara's guess (all zeros)
    V = np.zeros(len(states))
    
    while True:
        delta = 0
        # 2. Sweep through every state on the map
        for s in states:
            v_old = V[s]
            
            # 3. Calculate the Bellman Target
            # We look at every possible action and every possible outcome
            target = 0
            for a in actions:
                p_action = policy(s, a)
                for prob, s_next, reward in model(s, a):
                    target += p_action * prob * (reward + gamma * V[s_next])
            
            # 4. Learning happens here!
            # Because we have a perfect map, we don't need to 'smooth' our learning.
            # Effectively: V[s] = V[s] + 1.0 * (target - V[s])
            V[s] = target
            
            delta = max(delta, abs(v_old - V[s]))
            
        # 5. Stop when the truth has rippled through the whole map (convergence)
        if delta < theta:
            break
            
    return V
```

**Why this isn't "Machine Learning" in the traditional sense:**
In this code, there are no "samples" and no "experience." Sara is simply performing **numerical optimization** to solve her system of equations. The "learning" is the convergence of her guesses toward the fixed point.

**Key Insights:**
*   **Full Backup:** Unlike interacting where she sees one path, here she looks at *every* possible future state the map says is possible.
*   **Synchronous vs. Asynchronous:** Sara can update all states at once (as in the code above with `V[s] = target`) or update them in-place. Both will eventually find the truth.