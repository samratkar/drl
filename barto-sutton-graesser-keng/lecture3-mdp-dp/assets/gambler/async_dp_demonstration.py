import numpy as np
import matplotlib.pyplot as plt

def async_value_iteration(p_h=0.4, goal=100, theta=1e-10):
    """
    Asynchronous (In-Place) Value Iteration for the Gambler's Problem.
    
    In Async DP, we don't wait for a full sweep to complete before using
    updated values. We update states in-place and immediately use the new
    estimates for subsequent state updates in the same "sweep".
    """
    # States: 0, 1, ..., 100
    V = np.zeros(goal + 1)
    V[goal] = 1.0  # Terminal win state
    
    # We'll track the order of updates to show asynchrony
    # In a real async system, this might be random or prioritized
    state_indices = list(range(1, goal))
    
    sweeps = 0
    while True:
        delta = 0
        
        # Shuffle states to demonstrate that order doesn't break convergence
        # np.random.shuffle(state_indices) 
        
        for s in state_indices:
            v_old = V[s]
            
            # Possible stakes: 1 to min(s, goal - s)
            stakes = range(1, min(s, goal - s) + 1)
            
            # Bellman Optimality Update (In-Place)
            # Notice we read from and write to the SAME array 'V'
            best_val = 0
            for a in stakes:
                # This uses the MOST RECENT values of V[s+a] and V[s-a]
                # even if they were updated earlier in this very loop!
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

if __name__ == "__main__":
    V, policy, sweeps = async_value_iteration()
    print(f"Async Value Iteration converged in {sweeps} sweeps.")
    print(f"Win Probability at 50: {V[50]:.4f}")
    
    # Quick plot to verify results match synchronous version
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(V); plt.title('Async Value Function')
    plt.subplot(1, 2, 2); plt.bar(range(101), policy); plt.title('Async Policy')
    plt.show()
