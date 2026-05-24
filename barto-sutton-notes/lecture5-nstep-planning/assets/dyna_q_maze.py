import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Maze Environment (6x9)
HEIGHT = 6
WIDTH = 9
START = (2, 0)
GOAL = (0, 8)
WALLS = [(1, 2), (2, 2), (3, 2), (0, 7), (1, 7), (2, 7), (4, 5)]

ACTIONS = [0, 1, 2, 3] # UP, DOWN, LEFT, RIGHT

def step(state, action):
    r, c = state
    if action == 0: r_next, c_next = max(r - 1, 0), c
    elif action == 1: r_next, c_next = min(r + 1, HEIGHT - 1), c
    elif action == 2: r_next, c_next = r, max(c - 1, 0)
    elif action == 3: r_next, c_next = r, min(c + 1, WIDTH - 1)
    
    if (r_next, c_next) in WALLS:
        return state, 0
    if (r_next, c_next) == GOAL:
        return (r_next, c_next), 1
    return (r_next, c_next), 0

def dyna_q(n_planning, alpha=0.1, epsilon=0.1, episodes=50):
    Q = np.zeros((HEIGHT, WIDTH, 4))
    model = {} # (s, a) -> (s', r)
    steps_per_episode = []
    
    for _ in range(episodes):
        state = START
        steps = 0
        while state != GOAL:
            action = np.argmax(Q[state[0], state[1]]) if np.random.rand() > epsilon else np.random.choice(ACTIONS)
            next_state, reward = step(state, action)
            
            # Q-learning update
            Q[state[0], state[1], action] += alpha * (reward + np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
            
            # Model update
            model[(state, action)] = (next_state, reward)
            
            # Planning
            for _ in range(n_planning):
                # Sample a previously observed state-action pair
                idx = np.random.randint(len(model))
                (s_plan, a_plan), (s_next_plan, r_plan) = list(model.items())[idx]
                Q[s_plan[0], s_plan[1], a_plan] += alpha * (r_plan + np.max(Q[s_next_plan[0], s_next_plan[1]]) - Q[s_plan[0], s_plan[1], a_plan])
            
            state = next_state
            steps += 1
        steps_per_episode.append(steps)
    return steps_per_episode

if __name__ == "__main__":
    n_values = [0, 5, 50]
    episodes = 50
    runs = 30
    
    plt.figure(figsize=(10, 6))
    for n in n_values:
        results = np.zeros(episodes)
        for _ in tqdm(range(runs), desc=f'n={n}'):
            results += dyna_q(n, episodes=episodes)
        plt.plot(results / runs, label=f'{n} planning steps')
    
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.title('Figure 8.2: Dyna-Q on a simple Maze')
    plt.legend()
    plt.show()
