import numpy as np
import matplotlib.pyplot as plt

# Maximization Bias MDP (Example 6.7)
# States: A, B, Terminal
# Actions from A: Right (-> Terminal, R=0), Left (-> B, R=0)
# Actions from B: Many actions (-> Terminal, R ~ N(-0.1, 1))

STATE_A = 0
STATE_B = 1
TERMINAL = 2

ACTIONS_A = [0, 1] # 0: Left, 1: Right
ACTIONS_B = np.arange(10) # 10 actions

def step(state, action):
    if state == STATE_A:
        if action == 0: # Left
            return STATE_B, 0
        else: # Right
            return TERMINAL, 0
    else: # STATE_B
        return TERMINAL, np.random.normal(-0.1, 1.0)

def epsilon_greedy(Q, state, actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        # Find max actions and break ties randomly
        q_values = [Q[state, a] for a in actions]
        max_q = np.max(q_values)
        best_actions = [a for i, a in enumerate(actions) if q_values[i] == max_q]
        return np.random.choice(best_actions)

def q_learning(episodes=300, alpha=0.1, epsilon=0.1, gamma=1.0):
    Q = {}
    for a in ACTIONS_A: Q[(STATE_A, a)] = 0.0
    for a in ACTIONS_B: Q[(STATE_B, a)] = 0.0
    
    left_counts = []
    for _ in range(episodes):
        state = STATE_A
        left_selected = False
        while state != TERMINAL:
            actions = ACTIONS_A if state == STATE_A else ACTIONS_B
            action = epsilon_greedy(Q, state, actions, epsilon)
            
            if state == STATE_A and action == 0:
                left_selected = True
            
            next_state, reward = step(state, action)
            
            # Target
            if next_state == TERMINAL:
                target = reward
            else:
                next_actions = ACTIONS_A if next_state == STATE_A else ACTIONS_B
                target = reward + gamma * np.max([Q[(next_state, a)] for a in next_actions])
            
            Q[(state, action)] += alpha * (target - Q[(state, action)])
            state = next_state
            
        left_counts.append(1 if left_selected else 0)
    return left_counts

def double_q_learning(episodes=300, alpha=0.1, epsilon=0.1, gamma=1.0):
    Q1 = {}
    Q2 = {}
    for a in ACTIONS_A: Q1[(STATE_A, a)] = Q2[(STATE_A, a)] = 0.0
    for a in ACTIONS_B: Q1[(STATE_B, a)] = Q2[(STATE_B, a)] = 0.0
    
    left_counts = []
    for _ in range(episodes):
        state = STATE_A
        left_selected = False
        while state != TERMINAL:
            actions = ACTIONS_A if state == STATE_A else ACTIONS_B
            
            # Combine Q1 and Q2 for action selection
            Q_combined = {}
            for a in actions:
                Q_combined[(state, a)] = Q1[(state, a)] + Q2[(state, a)]
            
            action = epsilon_greedy(Q_combined, state, actions, epsilon)
            
            if state == STATE_A and action == 0:
                left_selected = True
            
            next_state, reward = step(state, action)
            
            if np.random.rand() < 0.5:
                # Update Q1
                if next_state == TERMINAL:
                    target = reward
                else:
                    next_actions = ACTIONS_A if next_state == STATE_A else ACTIONS_B
                    # Find best action using Q1
                    q1_values = [Q1[(next_state, a)] for a in next_actions]
                    best_a = next_actions[np.argmax(q1_values)]
                    target = reward + gamma * Q2[(next_state, best_a)]
                Q1[(state, action)] += alpha * (target - Q1[(state, action)])
            else:
                # Update Q2
                if next_state == TERMINAL:
                    target = reward
                else:
                    next_actions = ACTIONS_A if next_state == STATE_A else ACTIONS_B
                    # Find best action using Q2
                    q2_values = [Q2[(next_state, a)] for a in next_actions]
                    best_a = next_actions[np.argmax(q2_values)]
                    target = reward + gamma * Q1[(next_state, best_a)]
                Q2[(state, action)] += alpha * (target - Q2[(state, action)])
                
            state = next_state
            
        left_counts.append(1 if left_selected else 0)
    return left_counts

if __name__ == "__main__":
    print("Running Maximization Bias experiment (Q-learning vs Double Q-learning)...")
    runs = 1000
    episodes = 300
    q_left = np.zeros(episodes)
    dq_left = np.zeros(episodes)
    
    for r in tqdm(range(runs)):
        q_left += q_learning(episodes)
        dq_left += double_q_learning(episodes)
        
    q_left /= runs
    dq_left /= runs
    
    plt.plot(q_left, label='Q-learning')
    plt.plot(dq_left, label='Double Q-learning')
    plt.axhline(y=0.05, color='gray', linestyle='--', label='Optimal (left is 5% epsilon-greedy)')
    plt.xlabel('Episodes')
    plt.ylabel('% Left actions from A')
    plt.title('Maximization Bias (Example 6.7)')
    plt.legend()
    plt.show()
