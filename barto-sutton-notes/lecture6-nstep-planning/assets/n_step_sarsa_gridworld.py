import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Gridworld Environment
HEIGHT = 4
WIDTH = 12
START = (3, 0)
GOAL = (3, 11)
ACTIONS = [0, 1, 2, 3] # UP, DOWN, LEFT, RIGHT

def step(state, action):
    r, c = state
    if action == 0: r = max(r - 1, 0)
    elif action == 1: r = min(r + 1, HEIGHT - 1)
    elif action == 2: c = max(c - 1, 0)
    elif action == 3: c = min(c + 1, WIDTH - 1)
    
    if (r, c) == GOAL: return GOAL, 0
    return (r, c), -1

def n_step_sarsa(n, alpha, epsilon=0.1, episodes=50):
    Q = np.zeros((HEIGHT, WIDTH, 4))
    steps_per_episode = []
    
    for _ in range(episodes):
        state = START
        action = np.argmax(Q[state[0], state[1]]) if np.random.rand() > epsilon else np.random.choice(ACTIONS)
        
        states = [state]
        actions = [action]
        rewards = [0]
        
        T = float('inf')
        t = 0
        while True:
            if t < T:
                next_state, reward = step(states[t], actions[t])
                states.append(next_state)
                rewards.append(reward)
                if next_state == GOAL:
                    T = t + 1
                else:
                    next_action = np.argmax(Q[next_state[0], next_state[1]]) if np.random.rand() > epsilon else np.random.choice(ACTIONS)
                    actions.append(next_action)
            
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += rewards[i]
                if tau + n < T:
                    G += Q[states[tau + n][0], states[tau + n][1], actions[tau + n]]
                
                Q[states[tau][0], states[tau][1], actions[tau]] += alpha * (G - Q[states[tau][0], states[tau][1], actions[tau]])
            
            if tau == T - 1:
                break
            t += 1
        steps_per_episode.append(t)
    return steps_per_episode

if __name__ == "__main__":
    n_values = [1, 4, 16]
    alpha = 0.5
    episodes = 50
    runs = 50
    
    plt.figure(figsize=(10, 6))
    for n in n_values:
        results = np.zeros(episodes)
        for _ in tqdm(range(runs), desc=f'n={n}'):
            results += n_step_sarsa(n, alpha, episodes=episodes)
        plt.plot(results / runs, label=f'n={n}')
    
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.title('Figure 7.4: n-step Sarsa speed-up on Gridworld')
    plt.legend()
    plt.show()
