import numpy as np
import matplotlib.pyplot as plt

# Cliff Walking Gridworld
# 4 rows x 12 columns
# Start (3, 0), Goal (3, 11)
# Cliff: (3, 1) to (3, 10)
HEIGHT = 4
WIDTH = 12
START = (3, 0)
GOAL = (3, 11)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

def step(state, action):
    r, c = state
    if action == UP:
        next_state = (max(r - 1, 0), c)
    elif action == DOWN:
        next_state = (min(r + 1, HEIGHT - 1), c)
    elif action == LEFT:
        next_state = (r, max(c - 1, 0))
    elif action == RIGHT:
        next_state = (r, min(c + 1, WIDTH - 1))
    
    # Reward and Cliff check
    if next_state[0] == 3 and 1 <= next_state[1] <= 10:
        return START, -100
    elif next_state == GOAL:
        return next_state, 0
    else:
        return next_state, -1

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return np.argmax(Q[state[0], state[1]])

def sarsa(episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0):
    Q = np.zeros((HEIGHT, WIDTH, 4))
    rewards = []
    for _ in range(episodes):
        state = START
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        while state != GOAL:
            next_state, reward = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            # Q(S,A) <- Q(S,A) + alpha * [R + gamma*Q(S',A') - Q(S,A)]
            Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
            state = next_state
            action = next_action
            total_reward += reward
        rewards.append(total_reward)
    return Q, rewards

def q_learning(episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0):
    Q = np.zeros((HEIGHT, WIDTH, 4))
    rewards = []
    for _ in range(episodes):
        state = START
        total_reward = 0
        while state != GOAL:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward = step(state, action)
            # Q(S,A) <- Q(S,A) + alpha * [R + gamma*max_a Q(S',a) - Q(S,A)]
            Q[state[0], state[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return Q, rewards

if __name__ == "__main__":
    print("Running Cliff Walking experiment (Sarsa vs Q-learning)...")
    runs = 10
    sarsa_rewards = np.zeros(500)
    q_learning_rewards = np.zeros(500)
    
    for r in range(runs):
        _, s_rew = sarsa()
        _, q_rew = q_learning()
        sarsa_rewards += s_rew
        q_learning_rewards += q_rew
        
    sarsa_rewards /= runs
    q_learning_rewards /= runs
    
    plt.plot(sarsa_rewards, label='Sarsa')
    plt.plot(q_learning_rewards, label='Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.title('Sarsa vs Q-learning on Cliff Walking')
    plt.legend()
    plt.show()
