---
tags : [drl-class-notes]
title : "004 - Sara's sojourn - Temporal Difference"
classes : ["2026-05-16"]
---

## The Step-by-Step Instinct (Temporal Difference)

Waiting until nightfall by the Great Waterfall had its flaws. If Sara stepped onto thin ice, she shouldn't have to wait until she fell in to realize it was a bad idea. 

She needed a way to learn *while* walking. She combined the "guessing" of her campfire map days (Dynamic Programming) with the "actual walking" of her mapless days (Monte Carlo). She called this new instinct **Temporal Difference (TD) Learning**.

### 1. Bootstrapping on the Go

In Monte Carlo, Sara updated her memory using the true, final return ($G_t$).
In TD, she updates her memory after a **single step**. She takes a step, receives an immediate reward ($R_{t+1}$), and then looks at the *next* patch of ice ($S_{t+1}$). She uses her *current guess* of how good that next patch is to update her opinion of the patch she just left.

**The TD Update Rule:**
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD Target}} - V(S_t) \right]$$

*   **TD Target:** Her new, slightly better guess, formed by one real step ($R_{t+1}$) plus her old guess of the future ($\gamma V(S_{t+1})$).
*   **Bootstrapping:** Because she is using $V(S_{t+1})$ (a guess) to update $V(S_t)$ (another guess), she is "pulling herself up by her bootstraps."

### 2. SARSA: On-Policy Learning (The Cautious Walk)

Sara wanted to learn the value of the path she was *actually walking*, including all her random exploratory stumbles. 

**SARSA** stands for **State, Action, Reward, State, Action**.
It describes exactly how she learns:
1. She is in a **State** ($S$).
2. She chooses an **Action** ($A$) using $\epsilon$-greedy.
3. She gets a **Reward** ($R$) and lands in a new **State** ($S'$).
4. She chooses her next **Action** ($A'$) using $\epsilon$-greedy.
5. She updates her memory:

$$Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma Q(S', A') - Q(S, A) ]$$

Because SARSA evaluates the exact $\epsilon$-greedy policy Sara is currently using, it tends to learn a **safer** path. It knows Sara might randomly stumble, so it stays further away from holes.

### 3. Q-Learning: Off-Policy Learning (The Optimistic Walk)

Sara realized she could be bolder. What if she evaluated the *best possible* path, even if she was currently stumbling around exploring?

This is **Q-Learning**, an "Off-Policy" algorithm. It learns the optimal policy independently of the actions she is actually taking.
Instead of using the $Q$-value of the action she *will* take ($A'$), she uses the $Q$-value of the *best possible* action she *could* take:

$$Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma \max_a Q(S', a) - Q(S, A) ]$$

**The Key Difference:** The $\max_a$ operator.
Q-Learning is highly optimistic. It assumes, "Even if I'm exploring right now, when it comes time to really perform, I'll take the absolute best action." It tends to learn a faster, but **riskier** path right next to the holes.

### 4. The Algorithm in Python

```python
import numpy as np

def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in range(episodes):
        s, _ = env.reset()
        # Choose A from S
        a = epsilon_greedy(Q, s, epsilon, env.action_space.n)
        
        while True:
            # Take A, observe R, S'
            s_prime, r, terminated, truncated, _ = env.step(a)
            # Choose A' from S'
            a_prime = epsilon_greedy(Q, s_prime, epsilon, env.action_space.n)
            
            # SARSA Update
            td_target = r + gamma * Q[s_prime, a_prime] * (not terminated)
            Q[s, a] += alpha * (td_target - Q[s, a])
            
            s, a = s_prime, a_prime
            if terminated or truncated: break
    return Q

def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in range(episodes):
        s, _ = env.reset()
        
        while True:
            # Choose A from S
            a = epsilon_greedy(Q, s, epsilon, env.action_space.n)
            
            # Take A, observe R, S'
            s_prime, r, terminated, truncated, _ = env.step(a)
            
            # Q-Learning Update (The MAX operator)
            td_target = r + gamma * np.max(Q[s_prime]) * (not terminated)
            Q[s, a] += alpha * (td_target - Q[s, a])
            
            s = s_prime
            if terminated or truncated: break
    return Q
```

### 5. Comparing the Three Pillars

Sara now had three distinct ways to survive the FrozenLake:

| Feature | DP (Campfire) | MC (End of Day) | TD (Step-by-Step) |
| :--- | :--- | :--- | :--- |
| **Needs Map?** | Yes (Model-Based) | No (Model-Free) | No (Model-Free) |
| **Bootstraps?** | Yes | No | Yes |
| **Update Time** | Anytime | End of Episode | Every Step |
| **Variance** | Zero | High | Low |

By combining the real-world experience of MC with the fast, step-by-step guessing of DP, Sara found her ultimate survival instinct: **Temporal Difference Learning**.

*Next: But what happens when the lake grows so vast she can no longer memorize every step? The Great Fog descends, and Sara must build a Brain...*
