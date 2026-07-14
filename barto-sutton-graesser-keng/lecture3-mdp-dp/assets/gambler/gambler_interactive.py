# %% [markdown]
# # Case Study: The Gambler's Problem (Interactive Script)
#
# This script uses VS Code's "Interactive" mode (using `# %%` markers).
# It allows you to run cells individually without the service worker issues 
# of standard .ipynb files.
#
# ### Problem Description
# - **Goal**: Reach $100 capital.
# - **Dynamics**: A coin flip with probability $p_h$ of heads (win stake) and $1-p_h$ of tails (lose stake).
# - **States**: Capital $s$ from 1 to 99. 0 and 100 are terminal.
# - **Actions**: Stake $a$ from 1 to min(s, 100-s).

# %%
import numpy as np
import matplotlib.pyplot as plt

def solve_gambler(p_h=0.4, goal=100, theta=1e-10):
    # States: 0, 1, ..., 100
    V = np.zeros(goal + 1)
    V[goal] = 1.0
    
    # Value Iteration
    sweeps = 0
    while True:
        delta = 0
        for s in range(1, goal):
            v_old = V[s]
            stakes = range(1, min(s, goal - s) + 1)
            
            best_val = 0
            for a in stakes:
                val = p_h * V[s + a] + (1 - p_h) * V[s - a]
                if val > best_val:
                    best_val = val
            
            V[s] = best_val
            delta = max(delta, abs(v_old - V[s]))
        
        sweeps += 1
        if delta < theta:
            break
            
    # Policy Extraction
    policy = np.zeros(goal + 1)
    for s in range(1, goal):
        stakes = range(1, min(s, goal - s) + 1)
        action_values = [p_h * V[s+a] + (1-p_h) * V[s-a] for a in stakes]
        policy[s] = stakes[np.argmax(np.round(action_values, 10))]
        
    return V, policy, sweeps

# %% [markdown]
# ## Run Optimization
# We use $p_h = 0.4$ (unfair coin).

# %%
p_h = 0.4
V, policy, sweeps = solve_gambler(p_h=p_h)
print(f"Converged in {sweeps} sweeps.")
print(f"Win probability at $50 capital: {V[50]:.4f}")

# %% [markdown]
# ## Visualization
# The **Value Function** and **Optimal Policy**.

# %%
plt.figure(figsize=(12, 5))

# Value Function
plt.subplot(1, 2, 1)
plt.plot(range(101), V)
plt.title(f'Value Function (p_h={p_h})')
plt.xlabel('Capital')
plt.ylabel('Win Prob')
plt.grid(True, alpha=0.3)

# Optimal Policy
plt.subplot(1, 2, 2)
plt.bar(range(101), policy, color='orange')
plt.title('Optimal Policy (Stakes)')
plt.xlabel('Capital')
plt.ylabel('Stake')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
