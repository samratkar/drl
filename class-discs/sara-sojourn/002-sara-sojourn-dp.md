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

### 6. The Algorithms in Python

Here is how Sara’s campfire calculations look in code. Compare how **Policy Iteration** (the formal way) and **Value Iteration** (the shortcut) handle the search for the truth.

```python
import numpy as np

# --- 1. POLICY ITERATION (The Formal Way) ---
# Strategy: "Evaluate a plan perfectly before trying to improve it."
def policy_iteration(states, actions, model, gamma=0.9):
    # Start with a random plan (policy)
    policy = {s: np.random.choice(actions) for s in states}
    V = np.zeros(len(states))
    
    while True:
        # STEP A: Policy Evaluation (The "Soaking" Process)
        # We find the TRUE values for this SPECIFIC fixed policy.
        # This requires its own 'while' loop to let the truth ripple through.
        while True:
            delta = 0
            for s in states:
                v_old = V[s]
                # Calculate value based on current fixed policy[s]
                a = policy[s]
                # The Bellman Expectation Equation:
                V[s] = sum(prob * (reward + gamma * V[s_next]) 
                           for prob, s_next, reward in model(s, a))
                delta = max(delta, abs(v_old - V[s]))
            if delta < 1e-6: break # Stop when V perfectly matches this policy

        # STEP B: Policy Improvement (The "Greedy" Choice)
        # Now that we know the value of our current plan, we look for a better one.
        policy_stable = True
        for s in states:
            old_action = policy[s]
            
            # We calculate Q(s, a) for EVERY possible action 'a'.
            # Q(s, a) = "If I take action 'a' now, what is my expected return?"
            action_values = []
            for a in actions:
                # This is the 'Decision Bridge' Q(s, a)
                q_value = sum(prob * (reward + gamma * V[s_next]) 
                              for prob, s_next, reward in model(s, a))
                action_values.append(q_value)
            
            # GREEDY STEP: Pick the action with the highest Q-value.
            # We assume our current V is 'truth' and pick the best path forward.
            policy[s] = actions[np.argmax(action_values)]
            
            if old_action != policy[s]:
                policy_stable = False # If we found a better action, we aren't done!
        
        # If we didn't change any actions, we have reached the Optimal Policy.
        if policy_stable: break
    return policy, V

# --- 2. VALUE ITERATION (The Shortcut) ---
# Strategy: "Don't wait for perfection. Update values using the best action immediately."
def value_iteration(states, actions, model, gamma=0.9):
    V = np.zeros(len(states))
    
    while True:
        delta = 0
        # Instead of evaluating one policy, we merge evaluation and improvement.
        for s in states:
            v_old = V[s]
            
            # We calculate Q(s, a) for all actions, just like before...
            action_values = [sum(prob * (reward + gamma * V[s_next]) 
                             for prob, s_next, reward in model(s, a)) 
                             for a in actions]
            
            # THE SHORTCUT: Instead of V being tied to one policy,
            # we immediately set V to the MAX Q-value we found.
            # This is the Bellman Optimality Equation in action.
            V[s] = max(action_values) 
            
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < 1e-6: break # Stop when the optimal values stabilize
    
    # Once V is optimal, the policy is just the 'greedy' path through those values.
    policy = {s: actions[np.argmax([sum(prob * (reward + gamma * V[s_next]) 
              for prob, s_next, reward in model(s, a)) for a in actions])] 
              for s in states}
    return policy, V
```

### 10. How are they different?

While both find the same "Optimal Truth," they take different paths:

| Feature | Policy Iteration | Value Iteration |
| :--- | :--- | :--- |
| **Strategy** | Evaluates a policy perfectly before changing it. | Changes the value based on the "best guess" immediately. |
| **Loops** | **Nested:** Outer loop (Improvement) + Inner loop (Evaluation). | **Single:** One main loop that combines both. |
| **Speed** | Usually takes fewer *outer* iterations but more total work. | Takes more iterations but each step is simpler. |
| **Philosopy** | "Let's fully understand our current plan before making a change." | "Why wait? If I see a better action, I'll update my value right now." |

### 11. $Q(s, a)$: The Decision Bridge

In the code, you saw a complex line calculating `action_values`. This is the most important calculation in all of RL: the **Action-Value Function**, or **$Q(s, a)$**.

While $V(s)$ tells Sara how good a state is, it doesn't tell her *what to do*. To make a choice, she needs to know the value of her actions. 

**The $Q$ Calculation:**
```python
# Q(s, a) = Expected Return of taking action 'a' in state 's'
Q_s_a = sum(prob * (reward + gamma * V[s_next]) for prob, s_next, reward in model(s, a))
```

**How the "Greedy" Choice is made:**
To pick an action, Sara calculates $Q$ for every possible move and stores them in a list.
1.  **Look Ahead:** "If I go North, my $Q$ is 10. If I go South, my $Q$ is 15."
2.  **The Greedy Pick:** `np.argmax([10, 15])` $\to$ **South**.

This is called **Greedy** because Sara assumes her current estimates ($V$) are perfectly correct and chooses the single best path without hesitation.

### 12. Which one should Sara choose?

Both algorithms find the same optimal policy, but Sara chooses between them based on her "computing power" and the map size.

#### Use Policy Iteration when:
*   **The Map is Small:** If you can afford the nested loops, Policy Iteration often finds the *exact* optimal policy in very few "outer" steps.
*   **Actions are Limited:** If there are only a few possible actions, the "ladder-climbing" improvement step is very fast.
*   **You Need a Perfect Plan:** Because it evaluates the policy fully in every step, it is mathematically very stable.

#### Use Value Iteration when:
*   **The Map is Massive:** Since it only has one main loop, it is often much faster to compute for large state spaces.
*   **You are in a Hurry:** You don't "waste time" perfectly evaluating a bad policy. If you see a better action, you immediately pivot your value estimate toward it.
*   **The Map is Simple:** For most basic grid-worlds or puzzles, Value Iteration is the industry standard because it is simpler to code and usually converges faster in terms of raw CPU time.

### 13. The Tyranny of Greed

You are absolutely right: **Both algorithms are 100% greedy.** 

In Dynamic Programming, Sara assumes her "Perfect Map" (the Model) and her current "Value Estimates" are the source of truth. She never "explores" or "tries things out" to see what happens. She simply follows the math.

However, they apply this greediness at different times:

*   **In Policy Iteration:** Greed is a **specific step**. Sara spends a long time calculating the "Expectation" (averaging) for her current plan. Only after she is 100% sure of the values does she allow herself to be greedy and change her actions (**Policy Improvement**).
*   **In Value Iteration:** Greed is the **engine**. Sara is greedy in every single calculation. Every time she looks at a state, she immediately asks "What's the max?" and updates her value based only on that best option. 

**Why Greed works here (but fails in the real world):**
In DP, greed works because Sara has a **Perfect Map**. In the "real world" (where she doesn't have a map), being 100% greedy is dangerous because she might miss a hidden treasure just because her current guess says it's not there. But here, by the campfire, she can see everything—so greed is the most efficient path to the truth.

### 14. A Note on "Curiosity" (Why no $\epsilon$-greedy?)

You might have heard of **$\epsilon$-greedy**, where an agent sometimes takes a random action to "explore." 

In Dynamic Programming, Sara **never** uses $\epsilon$-greedy.
*   **No need to explore:** Exploration is for when you are lost in the dark. But Sara has a **Perfect Map**. She can "see" the result of every action without taking it. 
*   **Deterministic Planning:** When you have the truth, being random is just being wrong. If the math says North is 0.001% better than South, Sara will choose North 100% of the time.

$\epsilon$-greedy is a tool for **Model-Free RL**, where the agent has to walk through the world to learn it. For now, while Sara has her map, she is a perfect, cold-hearted calculator.

### Summary of the Sojourn so far:
*   **Dynamic Programming** is planning with a perfect map.
*   **Bellman Equations** define the relationship between states.
*   **$Q(s, a)$** is the bridge that turns state-values into action-decisions.
*   **Iteration** is the "soaking" process where the truth ripples through the map.
*   **Greedy Improvement** is picking the action with the highest $Q$ value.

**Key Insights:**
*   **The Target is Moving:** Every time Sara updates a state, she changes the target for all the states that lead into it.
*   **Bootstrapping:** We update a guess based on another guess.
*   **The "Max" is the Goal:** By switching from averaging (Expectation) to picking the best (Optimality), Sara finds the quickest path home.
*   **Full Backup:** She looks at *every* possible future state the map says is possible.
*   **Synchronous vs. Asynchronous:** Sara can update all states at once or update them in-place. Both will eventually find the truth.