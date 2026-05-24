import numpy as np
import matplotlib.pyplot as plt

# Random Walk Environment
# 0 (Term L), 1, 2, 3, 4, 5, 6 (Term R)
# Start at 3
STATES = np.arange(7)
START_STATE = 3
TRUE_VALUES = np.arange(1, 6) / 6.0 # [1/6, 2/6, 3/6, 4/6, 5/6] for states 1-5

def random_walk():
    state = START_STATE
    trajectory = [state]
    while state not in [0, 6]:
        if np.random.rand() < 0.5:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
    reward = 1 if state == 6 else 0
    return trajectory, reward

def td_0(V, alpha, trajectory, reward, gamma=1.0):
    for i in range(len(trajectory) - 1):
        s = trajectory[i]
        s_next = trajectory[i+1]
        # V(S_t) <- V(S_t) + alpha * [R + gamma * V(S_t+1) - V(S_t)]
        # R is 0 except at terminal transition
        r = reward if s_next == 6 else 0
        V[s] += alpha * (r + gamma * V[s_next] - V[s])
    return V

def mc_prediction(V, alpha, trajectory, reward):
    for s in trajectory[:-1]:
        # V(S_t) <- V(S_t) + alpha * [G - V(S_t)]
        V[s] += alpha * (reward - V[s])
    return V

def compute_rms_error(V):
    return np.sqrt(np.mean((V[1:6] - TRUE_VALUES)**2))

def experiment(episodes=100, runs=100):
    td_alphas = [0.05, 0.1, 0.15]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    
    results = {}

    for alpha in td_alphas:
        errors = np.zeros(episodes + 1)
        for _ in range(runs):
            V = np.ones(7) * 0.5
            V[0] = V[6] = 0
            errors[0] += compute_rms_error(V)
            for e in range(1, episodes + 1):
                traj, rew = random_walk()
                V = td_0(V, alpha, traj, rew)
                errors[e] += compute_rms_error(V)
        results[f'TD alpha={alpha}'] = errors / runs

    for alpha in mc_alphas:
        errors = np.zeros(episodes + 1)
        for _ in range(runs):
            V = np.ones(7) * 0.5
            V[0] = V[6] = 0
            errors[0] += compute_rms_error(V)
            for e in range(1, episodes + 1):
                traj, rew = random_walk()
                V = mc_prediction(V, alpha, traj, rew)
                errors[e] += compute_rms_error(V)
        results[f'MC alpha={alpha}'] = errors / runs
    
    return results

if __name__ == "__main__":
    print("Running Random Walk experiment (TD vs MC RMS error)...")
    results = experiment(100, 100)
    
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        ls = '-' if 'TD' in label else '--'
        plt.plot(data, label=label, linestyle=ls)
    
    plt.xlabel('Episodes')
    plt.ylabel('Empirical RMS error, averaged over states')
    plt.title('TD vs MC on Random Walk')
    plt.legend()
    plt.show()
