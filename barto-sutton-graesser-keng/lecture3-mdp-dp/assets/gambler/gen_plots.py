import numpy as np
import matplotlib.pyplot as plt

def solve_gambler(p_h=0.4, goal=100, theta=1e-10):
    V = np.zeros(goal + 1)
    V[goal] = 1.0
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
        if delta < theta:
            break
            
    policy = np.zeros(goal + 1)
    for s in range(1, goal):
        stakes = range(1, min(s, goal - s) + 1)
        action_values = [p_h * V[s+a] + (1-p_h) * V[s-a] for a in stakes]
        policy[s] = stakes[np.argmax(np.round(action_values, 10))]
        
    return V, policy

V, policy = solve_gambler()

plt.figure(figsize=(12, 5))

# Value Function
plt.subplot(1, 2, 1)
plt.plot(range(101), V)
plt.title('Value Function (Win Probability)')
plt.xlabel('Capital')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)

# Optimal Policy
plt.subplot(1, 2, 2)
plt.bar(range(101), policy, color='orange')
plt.title('Optimal Policy (Stakes)')
plt.xlabel('Capital')
plt.ylabel('Stake')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gambler_results.png')
