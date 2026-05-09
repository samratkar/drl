import numpy as np
import matplotlib.pyplot as plt

def gambler_problem(p_h=0.4, goal=100, theta=1e-9):
    """
    Solves the Gambler's Problem using Value Iteration.
    
    Args:
        p_h: Probability of the coin coming up heads.
        goal: The target capital to reach.
        theta: Convergence threshold.
    """
    # States: 0, 1, ..., 100. 0 and 100 are terminal.
    V = np.zeros(goal + 1)
    V[goal] = 1.0  # Reward is +1 for reaching the goal
    
    # 1. Value Iteration
    while True:
        delta = 0
        for s in range(1, goal):  # For each capital from 1 to 99
            v_old = V[s]
            
            # Possible stakes: 1 to min(s, goal - s)
            # Note: stake 0 is omitted as it leads to no change
            stakes = range(1, min(s, goal - s) + 1)
            action_values = []
            
            for a in stakes:
                # Bellman Equation: p_h * V[s+a] + (1-p_h) * V[s-a]
                # Reward is only at the end (captured in V[goal]=1)
                val = p_h * V[s + a] + (1 - p_h) * V[s - a]
                action_values.append(val)
            
            V[s] = max(action_values)
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break
            
    # 2. Extract optimal policy
    policy = np.zeros(goal + 1)
    for s in range(1, goal):
        stakes = range(1, min(s, goal - s) + 1)
        action_values = []
        for a in stakes:
            val = p_h * V[s + a] + (1 - p_h) * V[s - a]
            # Use rounding to handle floating point precision issues in argmax
            action_values.append(round(val, 10))
            
        # The book shows one optimal policy, though there might be ties
        # We pick the smallest stake for consistency with standard plots
        best_stake = stakes[np.argmax(action_values)]
        policy[s] = best_stake
        
    return V, policy

def plot_results(V, policy, p_h):
    plt.figure(figsize=(12, 6))
    
    # Plot Value Function
    plt.subplot(1, 2, 1)
    plt.plot(V[:-1])
    plt.title(f'Value Function (p_h={p_h})')
    plt.xlabel('Capital')
    plt.ylabel('Value (Prob of winning)')
    plt.grid(True)
    
    # Plot Policy
    plt.subplot(1, 2, 2)
    plt.bar(range(len(policy)), policy)
    plt.title(f'Optimal Policy (Stake vs Capital)')
    plt.xlabel('Capital')
    plt.ylabel('Stake')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p_h = 0.4
    V, policy = gambler_problem(p_h=p_h)
    
    print("Optimization Complete.")
    print(f"Sample Values: V[25]={V[25]:.4f}, V[50]={V[50]:.4f}, V[75]={V[75]:.4f}")
    
    # Uncomment to show plots if running locally
    # plot_results(V, policy, p_h)
