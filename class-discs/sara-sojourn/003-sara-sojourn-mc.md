---
tags : [drl-class-notes]
title : "003 - Sara's sojourn - Monte Carlo"
classes : ["2026-05-09"]
---

## The Ash and the Abyss (Monte Carlo)

The fire was swift. As Sara slept by the Great Waterfall, a stray ember caught her pack. When she woke, her "perfect map"—the beautiful Model $\mathcal{P}$ she had so carefully calculated—was nothing but grey ash.

Sara is no longer a "Cold Calculator." She can no longer sit by the fire and predict the world. To learn, she must walk. She must fall. She must **Experience**.

### 1. Model-Free: Learning from the Dark

In Dynamic Programming, Sara knew the "Model." She knew that moving North had a 33% chance of success. Now, she knows nothing. This is **Model-Free Reinforcement Learning**.

Instead of solving equations *about* the world, Sara learns by **interacting** *with* the world. She takes an action, sees where she ends up, and remembers the result.

### 2. The Monte Carlo Way: "Wait for the End"

Sara decides on a new strategy: **Monte Carlo (MC)**. 
She won't update her memory after every step. Instead, she will walk a full **Episode** (from the Start to either the Goal or a Hole). Only when the journey is over—when she is either safe or wet—does she look back and calculate her total rewards (**The Return, $G_t$**).

**The MC Update Rule:**
$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

*   **$G_t$:** The actual, ground-truth total reward she received from time $t$ until the end.
*   **$V(s)$:** Her old guess.
*   **$\alpha$:** Her learning rate (how much she trusts this new journey).

### 3. No Bootstrapping (The "Anti-DP")

In DP, Sara updated her guess of $V(s)$ using her *other guess* of $V(s')$. This is called **Bootstrapping**.
In Monte Carlo, Sara is done with guesses. She uses the **Actual Return**. 
*   **DP:** "I think $s$ is good because I *think* $s'$ is good."
*   **MC:** "I know $s$ is good because I *actually walked* from $s$ to the goal."

### 4. The Problem of Exploration: $\epsilon$-greedy

Since Sara no longer has a map, she faces a dilemma:
1.  **Exploitation:** Follow the path she *thinks* is best.
2.  **Exploration:** Try a random path to see if there's a better way.

If she is 100% greedy (like she was with DP), she might find one mediocre path to the goal and walk it forever, never realizing there's a chest of gold just one step to the left. 

To solve this, Sara uses **$\epsilon$-greedy**:
*   With probability **$1 - \epsilon$**, she takes the best-known action.
*   With probability **$\epsilon$**, she takes a completely random action.

### 5. First-Visit vs. Every-Visit

If Sara walks in a circle ($A \to B \to A \to \text{Goal}$), she visits state $A$ twice. 
*   **First-Visit MC:** She only uses the return from the *first* time she saw $A$ in that episode.
*   **Every-Visit MC:** She updates her memory for $A$ *every* time she passes through it.

### 6. The Algorithm in Python

Here is how Sara learns to navigate the slippery ice without a map. Note how she must **generate a full episode** before she can update her $Q$-table.

```python
import numpy as np

def monte_carlo_control(env, episodes=5000, gamma=0.9, epsilon=0.1, alpha=0.01):
    # We now track Q(s, a) directly because we don't have a model to look ahead!
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for i in range(episodes):
        # 1. GENERATE AN EPISODE
        # Sara walks blindly until she reaches a terminal state.
        state, _ = env.reset()
        episode = []
        while True:
            # Epsilon-Greedy choice: Exploration vs Exploitation
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if terminated or truncated: break
            
        # 2. UPDATE FROM EXPERIENCE
        # The episode is over. Sara looks back at her tracks.
        G = 0
        visited_sa = set()
        for s, a, r in reversed(episode):
            G = r + gamma * G # Calculate the actual return
            
            # First-visit update
            if (s, a) not in visited_sa:
                # The Incremental Mean Update
                Q[s, a] += alpha * (G - Q[s, a])
                visited_sa.add((s, a))
                
    return Q
```

### 7. Why MC can be Dangerous (Variance)

Monte Carlo is "honest" because it uses actual returns, but it is **High Variance**. 
If the ice is very slippery, Sara might take the "perfect" action but still fall into a hole due to bad luck. If she only updates once at the end of the day, she might think her *action* was bad, when it was actually just the *wind*. 

Because an episode can be very long, there are many random things that can happen. This makes $G_t$ a very "noisy" signal. Sara needs many, many episodes to find the truth.

### 8. Comparing DP and MC

| Feature | Dynamic Programming (DP) | Monte Carlo (MC) |
| :--- | :--- | :--- |
| **Knowledge** | Needs a Perfect Map (Model). | Needs Zero Knowledge. |
| **Learning** | Learns by thinking (Planning). | Learns by doing (Experience). |
| **Updates** | After every "step" (Bootstraps). | Only at the end of the journey. |
| **Bias/Variance** | High Bias (Guesses are often wrong). | High Variance (Episodes are noisy). |
| **Complexity** | Small/Simple worlds. | Large/Unknown/Complex worlds. |

### Summary of the Sojourn:
*   **Model-Free** means learning from experience, not a map.
*   **Monte Carlo** updates based on the full journey's return.
*   **$\epsilon$-greedy** ensures Sara doesn't get stuck in a "local optimum" by forcing her to explore.
*   **No Bootstrapping** means MC is grounded in reality, but it is slow and noisy.

Sara has survived the loss of her map. By walking the ice, falling, and getting back up, she has learned something the math couldn't tell her: the *feeling* of the world. But she wonders... is there a way to learn *during* the walk, without waiting for the sun to set? 

*Next: Temporal Difference Learning (The Best of Both Worlds)...*
