import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Shortcut Maze Environment (6x9)
HEIGHT = 6
WIDTH = 9
START = (5, 3)
GOAL = (0, 8)
# Initial walls (blocking the shortcut)
WALLS_INITIAL = [(3, i) for i in range(0, 8)]
# Later walls (opening the shortcut)
WALLS_LATER = [(3, i) for i in range(1, 9)]

ACTIONS = [0, 1, 2, 3] # UP, DOWN, LEFT, RIGHT

def step(state, action, walls):
    r, c = state
    if action == 0: r_next, c_next = max(r - 1, 0), c
    elif action == 1: r_next, c_next = min(r + 1, HEIGHT - 1), c
    elif action == 2: r_next, c_next = r, max(c - 1, 0)
    elif action == 3: r_next, c_next = r, min(c + 1, WIDTH - 1)
    
    if (r_next, c_next) in walls:
        return state, 0
    if (r_next, c_next) == GOAL:
        return (r_next, c_next), 1
    return (r_next, c_next), 0

def dyna_q_experiment(method='Dyna-Q', n_planning=50, total_steps=6000, alpha=0.1, epsilon=0.1, kappa=1e-3):
    Q = np.zeros((HEIGHT, WIDTH, 4))
    model = {} # (s, a) -> (s', r, time_step)
    
    # Initialize model with all actions to 0 reward and same state for Dyna-Q+
    if method == 'Dyna-Q+':
        for r in range(HEIGHT):
            for c in range(WIDTH):
                for a in ACTIONS:
                    model[((r, c), a)] = ((r, c), 0, 0)
                    
    cumulative_rewards = []
    total_reward = 0
    state = START
    walls = WALLS_INITIAL
    
    for t in range(1, total_steps + 1):
        if t == 3000:
            walls = WALLS_LATER
            
        action = np.argmax(Q[state[0], state[1]]) if np.random.rand() > epsilon else np.random.choice(ACTIONS)
        next_state, reward = step(state, action, walls)
        total_reward += reward
        cumulative_rewards.append(total_reward)
        
        # Q-learning update
        Q[state[0], state[1], action] += alpha * (reward + np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
        
        # Model update
        model[(state, action)] = (next_state, reward, t)
        
        # Planning
        for _ in range(n_planning):
            # Sample a state-action pair
            if method == 'Dyna-Q':
                idx = np.random.randint(len(model))
                (s_plan, a_plan), (s_next_plan, r_plan, t_plan) = list(model.items())[idx]
            else: # Dyna-Q+
                # Randomly pick from ALL previously tried state-actions
                idx = np.random.randint(len(model))
                (s_plan, a_plan), (s_next_plan, r_plan, t_plan) = list(model.items())[idx]
                # Exploration bonus
                r_plan += kappa * np.sqrt(t - t_plan)
                
            Q[s_plan[0], s_plan[1], a_plan] += alpha * (r_plan + np.max(Q[s_next_plan[0], s_next_plan[1]]) - Q[s_plan[0], s_plan[1], a_plan])
            
        if next_state == GOAL:
            state = START
        else:
            state = next_state
            
    return cumulative_rewards

if __name__ == "__main__":
    total_steps = 6000
    runs = 10
    
    plt.figure(figsize=(10, 6))
    for method in ['Dyna-Q', 'Dyna-Q+']:
        results = np.zeros(total_steps)
        for _ in tqdm(range(runs), desc=method):
            results += dyna_q_experiment(method, total_steps=total_steps)
        plt.plot(results / runs, label=method)
    
    plt.axvline(x=3000, color='gray', linestyle='--', label='Shortcut opens')
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.title('Figure 8.5: Shortcut Maze (Dyna-Q vs Dyna-Q+)')
    plt.legend()
    plt.show()
