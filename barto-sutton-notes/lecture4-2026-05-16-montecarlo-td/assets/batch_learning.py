import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Random Walk Environment
STATES = np.arange(7) # 0, 1, 2, 3, 4, 5, 6
START_STATE = 3
TRUE_VALUES = np.arange(1, 6) / 6.0

def random_walk_episodes(n_episodes):
    episodes = []
    for _ in range(n_episodes):
        state = START_STATE
        trajectory = [state]
        while state not in [0, 6]:
            if np.random.rand() < 0.5: state -= 1
            else: state += 1
            trajectory.append(state)
        reward = 1 if state == 6 else 0
        episodes.append((trajectory, reward))
    return episodes

def batch_update(episodes, alpha, method='TD', gamma=1.0):
    V = np.ones(7) * 0.5
    V[0] = V[6] = 0
    
    # Iterate until convergence for the batch
    for _ in range(100): # max iterations for convergence
        v_update = np.zeros(7)
        for traj, reward in episodes:
            if method == 'TD':
                for i in range(len(traj) - 1):
                    s = traj[i]
                    s_next = traj[i+1]
                    r = reward if s_next == 6 else 0
                    v_update[s] += (r + gamma * V[s_next] - V[s])
            else: # MC
                for s in traj[:-1]:
                    v_update[s] += (reward - V[s])
        
        if np.sum(np.abs(v_update)) < 1e-3: break
        V += alpha * v_update
    return V

def compute_rms(V):
    return np.sqrt(np.mean((V[1:6] - TRUE_VALUES)**2))

def experiment(episodes_range, runs=100):
    td_errors = []
    mc_errors = []
    
    for n in tqdm(episodes_range):
        td_err_sum = 0
        mc_err_sum = 0
        for _ in range(runs):
            episodes = random_walk_episodes(n)
            
            V_td = batch_update(episodes, 0.01, method='TD')
            td_err_sum += compute_rms(V_td)
            
            V_mc = batch_update(episodes, 0.01, method='MC')
            mc_err_sum += compute_rms(V_mc)
            
        td_errors.append(td_err_sum / runs)
        mc_errors.append(mc_err_sum / runs)
        
    return td_errors, mc_errors

if __name__ == "__main__":
    episodes_range = [2, 5, 10, 20, 50, 100]
    td_err, mc_err = experiment(episodes_range, runs=50)
    
    plt.plot(episodes_range, td_err, label='Batch TD(0)', marker='o')
    plt.plot(episodes_range, mc_err, label='Batch MC', marker='s')
    plt.xlabel('Number of Episodes in Batch')
    plt.ylabel('RMS Error')
    plt.title('Figure 6.2: Batch Training TD vs MC')
    plt.legend()
    plt.show()
