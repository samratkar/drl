import numpy as np

# 1. First-visit MC Prediction
def first_visit_mc_prediction(env, policy, episodes, gamma=1.0):
    V = {}
    returns_sum = {}
    returns_count = {}
    
    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, reward))
            state = next_state
            
        G = 0
        visited_states = set()
        for s, r in reversed(trajectory):
            G = r + gamma * G
            if s not in visited_states:
                visited_states.add(s)
                returns_sum[s] = returns_sum.get(s, 0) + G
                returns_count[s] = returns_count.get(s, 0) + 1
                V[s] = returns_sum[s] / returns_count[s]
    return V

# 2. TD(0) Prediction
def td_0_prediction(env, policy, alpha, episodes, gamma=1.0):
    V = {} # Initialize with 0 for all encountered states
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            if s not in V: V[s] = 0
            a = policy(s)
            s_next, r, done, _ = env.step(a)
            if s_next not in V: V[s_next] = 0
            
            # TD update
            target = r + (gamma * V[s_next] if not done else 0)
            V[s] += alpha * (target - V[s])
            s = s_next
    return V

# 3. Sarsa Control
def sarsa_control(env, alpha, epsilon, episodes, gamma=1.0):
    Q = {} # (s, a) -> value
    for _ in range(episodes):
        s = env.reset()
        # Epsilon-greedy selection
        a = select_action(Q, s, epsilon, env.action_space.n)
        done = False
        while not done:
            s_next, r, done, _ = env.step(a)
            a_next = select_action(Q, s_next, epsilon, env.action_space.n)
            
            # Update
            q_next = Q.get((s_next, a_next), 0) if not done else 0
            target = r + gamma * q_next
            Q[(s, a)] = Q.get((s, a), 0) + alpha * (target - Q.get((s, a), 0))
            
            s, a = s_next, a_next
    return Q

# 4. Q-Learning Control
def q_learning_control(env, alpha, epsilon, episodes, gamma=1.0):
    Q = {}
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = select_action(Q, s, epsilon, env.action_space.n)
            s_next, r, done, _ = env.step(a)
            
            # Max update
            max_q_next = 0
            if not done:
                max_q_next = max([Q.get((s_next, act), 0) for act in range(env.action_space.n)])
            
            target = r + gamma * max_q_next
            Q[(s, a)] = Q.get((s, a), 0) + alpha * (target - Q.get((s, a), 0))
            s = s_next
    return Q

def select_action(Q, s, eps, n_actions):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    q_vals = [Q.get((s, a), 0) for a in range(n_actions)]
    return np.argmax(q_vals)
