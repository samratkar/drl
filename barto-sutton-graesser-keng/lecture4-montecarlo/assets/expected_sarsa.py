import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Cliff Walking Environment
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
    
    if r == 3 and 1 <= c <= 10: return START, -100
    if (r, c) == GOAL: return GOAL, 0
    return (r, c), -1

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon: return np.random.choice(ACTIONS)
    return np.argmax(Q[state[0], state[1]])

def run_episode(Q, alpha, epsilon, method='Sarsa'):
    state = START
    total_reward = 0
    if method == 'Sarsa':
        action = epsilon_greedy(Q, state, epsilon)
        while state != GOAL:
            next_state, reward = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state[0], state[1], action] += alpha * (reward + Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
            state, action = next_state, next_action
            total_reward += reward
    elif method == 'Q-learning':
        while state != GOAL:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward = step(state, action)
            Q[state[0], state[1], action] += alpha * (reward + np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
            state = next_state
            total_reward += reward
    elif method == 'Expected Sarsa':
        while state != GOAL:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward = step(state, action)
            # Expected Value
            q_next = Q[next_state[0], next_state[1]]
            best_a = np.argmax(q_next)
            expected_q = 0
            for a in ACTIONS:
                if a == best_a: expected_q += (1 - epsilon + epsilon/4) * q_next[a]
                else: expected_q += (epsilon/4) * q_next[a]
            
            Q[state[0], state[1], action] += alpha * (reward + expected_q - Q[state[0], state[1], action])
            state = next_state
            total_reward += reward
    return total_reward

def experiment(alphas, episodes=1000, runs=10):
    methods = ['Sarsa', 'Q-learning', 'Expected Sarsa']
    results = {m: [] for m in methods}
    
    for method in methods:
        for alpha in tqdm(alphas, desc=method):
            rewards = []
            for _ in range(runs):
                Q = np.zeros((HEIGHT, WIDTH, 4))
                run_rewards = []
                for _ in range(episodes):
                    run_rewards.append(run_episode(Q, alpha, 0.1, method))
                rewards.append(np.mean(run_rewards))
            results[method].append(np.mean(rewards))
    return results

if __name__ == "__main__":
    alphas = np.linspace(0.1, 1.0, 10)
    results = experiment(alphas, episodes=500, runs=10)
    
    for method, data in results.items():
        plt.plot(alphas, data, label=method, marker='o')
    
    plt.xlabel('Alpha')
    plt.ylabel('Average Reward per Episode')
    plt.title('Figure 6.4: Interim Performance on Cliff Walking')
    plt.legend()
    plt.show()
