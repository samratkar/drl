import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 19-state Random Walk (States 1-19, 0 and 20 terminal)
N_STATES = 19
START_STATE = 10
TERMINAL_L = 0
TERMINAL_R = 20
TRUE_VALUES = np.arange(-19, 21, 2) / 20.0
TRUE_VALUES = TRUE_VALUES[1:-1] # Values for states 1-19

def random_walk_step(state):
    if np.random.rand() < 0.5:
        return state - 1
    return state + 1

def n_step_td(n, alpha, episodes=10, gamma=1.0):
    V = np.zeros(21) # States 0-20
    errors = []
    
    for _ in range(episodes):
        states = [START_STATE]
        rewards = [0]
        T = float('inf')
        t = 0
        
        while True:
            if t < T:
                next_state = random_walk_step(states[t])
                states.append(next_state)
                reward = 1 if next_state == TERMINAL_R else (-1 if next_state == TERMINAL_L else 0)
                rewards.append(reward)
                if next_state in [TERMINAL_L, TERMINAL_R]:
                    T = t + 1
            
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma**(i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma**n) * V[states[tau + n]]
                
                V[states[tau]] += alpha * (G - V[states[tau]])
            
            if tau == T - 1:
                break
            t += 1
        
        # Calculate RMS error for states 1-19
        rms = np.sqrt(np.mean((V[1:20] - TRUE_VALUES)**2))
        errors.append(rms)
        
    return np.mean(errors)

if __name__ == "__main__":
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    alphas = np.linspace(0, 1, 21)
    episodes = 10
    runs = 100
    
    plt.figure(figsize=(10, 6))
    for n in n_values:
        results = []
        for alpha in alphas:
            err_sum = 0
            for _ in range(runs):
                err_sum += n_step_td(n, alpha, episodes)
            results.append(err_sum / runs)
        plt.plot(alphas, results, label=f'n={n}')
    
    plt.xlabel('Alpha')
    plt.ylabel(f'Average RMS error over first {episodes} episodes')
    plt.ylim([0, 0.55])
    plt.title('Figure 7.2: Performance of n-step TD on 19-state Random Walk')
    plt.legend()
    plt.show()
