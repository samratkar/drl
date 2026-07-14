import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Example 5.5: Infinite Variance
# Single state, action back with 0.9 (R=0), action terminal with 0.1 (R=1)
# Behavior policy: 0.5/0.5
# Target policy: always go back

def infinite_variance_experiment(episodes=1000000):
    ord_is_estimates = []
    ord_is_sum = 0
    
    for e in tqdm(range(episodes)):
        # Trajectory
        rho = 1.0
        while True:
            # Action 0: Back (prob 0.9), Action 1: Term (prob 0.1)
            action = np.random.choice([0, 1])
            # pi(0) = 1.0, pi(1) = 0.0
            # b(0) = 0.5, b(1) = 0.5
            if action == 0:
                rho *= (1.0 / 0.5)
                # Next state logic
                if np.random.rand() < 0.1: # Actually goes to terminal
                    reward = 0
                    break
            else: # action 1
                rho *= (0.0 / 0.5)
                reward = 1 # but target policy never takes this
                break
        
        # In this specific simple MDP from p. 106:
        # The ONLY trajectory with non-zero rho and non-zero reward is:
        # ALL actions are 'Back' (action 0) and then it happens to terminate.
        # But in the book's specific example:
        # "The target policy always takes the action that returns to the nonterminal state."
        # "The reward is 1 only when the episode ends... and 0 otherwise."
        
        # Corrected simulation for Example 5.5:
        # Actions: left (to terminal, R=1), right (back to state, R=0)
        # pi(right) = 1, b(right) = 0.5
        rho = 1.0
        reward = 0
        while True:
            action = np.random.choice(['left', 'right']) # behavior policy 50/50
            if action == 'right':
                rho *= (1.0 / 0.5)
                # stays in state
            else: # action == 'left'
                rho *= (0.0 / 0.5)
                reward = 1
                break
        
        estimate = rho * reward
        ord_is_sum += estimate
        if (e + 1) % 100 == 0:
            ord_is_estimates.append(ord_is_sum / (e + 1))
            
    return ord_is_estimates

if __name__ == "__main__":
    print("Running Infinite Variance experiment...")
    estimates = infinite_variance_experiment(100000)
    
    plt.plot(np.arange(len(estimates)) * 100, estimates)
    plt.xlabel('Episodes')
    plt.ylabel('Ordinary IS Estimate of V(s)')
    plt.title('Figure 5.5: Infinite Variance Example')
    plt.axhline(y=1, color='r', linestyle='--', label='True Value (if gamma=1 and it eventually terminates)')
    # Note: in the book example, the value is 1 if it eventually terminates.
    plt.show()
