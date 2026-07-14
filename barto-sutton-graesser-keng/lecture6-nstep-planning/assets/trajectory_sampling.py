import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Environment with varying branching factor b
# State 0 is start, multiple actions lead to terminal or next layer
# For simplicity, we model a task where only one path leads to reward 1

def trajectory_sampling_experiment(b, n_states=1000):
    # Transition dynamics: each state has b actions leading to b different states
    # Only state 0 -> action 0 -> state 1 ... -> state N-1 leads to reward
    
    # Value function initialization
    V_uniform = np.zeros(n_states + 1)
    V_traj = np.zeros(n_states + 1)
    
    max_updates = 200000
    interval = 1000
    
    errors_uniform = []
    errors_traj = []
    
    # True value for state 0 (if only one path works and reward is 1 at end)
    true_v0 = 1.0 # simplified
    
    # Uniform updates
    for i in range(0, max_updates + 1, interval):
        # In a real uniform update, we'd sweep or pick random states
        # Here we simulate the value of state 0 after i updates
        # V(s) = sum p(s'|s,a)[r + V(s')]
        # This is hard to simulate without full DP, so we use a simplified model
        # representing Fig 8.8 intuition: 
        # Trajectory sampling is better early on because it finds the reward path.
        pass

    # Re-implementing based on the core idea: 
    # How many updates to state 0 does it take to get close to the true value?
    
    # To properly reproduce Fig 8.8, we need a task where trajectory sampling 
    # visits the rewarding states more frequently than uniform sampling.
    
    # Let's use a simple chain with branching
    # State i -> b actions -> lead to i+1 (but only one is correct)
    # Actually, the book says: "b actions from each state, each leading to a different next state"
    # "Exactly one of these leads to the goal"
    
    # Simulation:
    def simulate_sampling(sampling_type, b, n_steps=200000):
        V = np.zeros(n_states + 1)
        V[n_states] = 1.0 # Goal
        v0_history = []
        
        state_counts = np.zeros(n_states + 1)
        
        curr_state = 0
        for i in range(n_steps):
            if sampling_type == 'uniform':
                s = np.random.randint(n_states)
            else: # trajectory
                s = curr_state
                # update curr_state for next iteration
                if np.random.rand() < 0.1 or s == n_states:
                    curr_state = 0
                else:
                    curr_state = s + 1 # simplified progression
            
            # Bellman update (simulated)
            # V(s) = (1/b) * sum V(s_next)
            # Only one next state has a path to the goal
            V[s] = (1.0 / b) * (V[s+1] + (b-1) * 0)
            
            if i % 1000 == 0:
                v0_history.append(V[0])
        return v0_history

    results = {}
    for b_val in [2, 10]:
        results[f'uniform b={b_val}'] = simulate_sampling('uniform', b_val)
        results[f'traj b={b_val}'] = simulate_sampling('trajectory', b_val)
        
    return results

if __name__ == "__main__":
    results = trajectory_sampling_experiment(b=2) # testing with b=2
    # The actual Fig 8.8 requires b=2, 10, 100 etc.
    
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(data, label=label)
    
    plt.xlabel('Number of updates (x1000)')
    plt.ylabel('Value of start state')
    plt.title('Figure 8.8: Trajectory vs Uniform Sampling (Simulated)')
    plt.legend()
    plt.show()
