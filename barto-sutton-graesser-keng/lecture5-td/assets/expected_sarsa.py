import numpy as np
import gymnasium as gym

class ExpectedSarsaAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table to zeros
        self.Q = np.zeros((num_states, num_actions))
        
    def get_action_probabilities(self, state):
        """
        Returns the action probabilities under the target epsilon-greedy policy.
        """
        q_values = self.Q[state]
        max_q = np.max(q_values)
        
        # Find all actions that achieve the max Q-value (handling ties)
        best_actions = np.flatnonzero(q_values == max_q)
        num_best = len(best_actions)
        
        # Epsilon-greedy distribution
        probabilities = np.full(self.num_actions, self.epsilon / self.num_actions)
        probabilities[best_actions] += (1.0 - self.epsilon) / num_best
        
        return probabilities
        
    def choose_action(self, state):
        """
        Choose action using the epsilon-greedy behavior policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            # Greedy action selection with random tie-breaking
            q_values = self.Q[state]
            max_q = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_q)
            if len(best_actions) == 1:
                return best_actions[0]
            return np.random.choice(best_actions)
            
    def update(self, state, action, reward, next_state, terminated):
        """
        Updates the action-value function Q using the Expected Sarsa update rule.
        """
        if terminated:
            # If the episode ended, there is no future expected value
            expected_value = 0.0
        else:
            # Expected Sarsa: sum of probabilities * Q-values for all next actions
            probs = self.get_action_probabilities(next_state)
            expected_value = np.sum(probs * self.Q[next_state])
            
        td_target = reward + self.gamma * expected_value
        td_error = td_target - self.Q[state, action]
        
        # Update action-value
        self.Q[state, action] += self.alpha * td_error

def train_agent(episodes=500, alpha=0.5, epsilon=0.1):
    # Load Cliff Walking gridworld environment from Gymnasium
    # State space: 48 states (4x12 grid)
    # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    env = gym.make('CliffWalking-v1')
    
    agent = ExpectedSarsaAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        alpha=alpha,
        gamma=1.0,
        epsilon=epsilon
    )
    
    print(f"Training Expected Sarsa Agent on CliffWalking-v0 for {episodes} episodes...")
    print(f"Parameters: alpha={alpha}, epsilon={epsilon}, gamma=1.0\n")
    
    rewards_history = []
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Expected Sarsa update
            agent.update(state, action, reward, next_state, terminated)
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        # Print progress every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode:03d}/{episodes} | Average Reward (last 50 ep): {avg_reward:.1f}")
            
    env.close()
    return agent, rewards_history

def print_learned_policy(agent):
    """
    Displays the learned greedy policy on a grid.
    Grid size for CliffWalking is 4 x 12.
    """
    height, width = 4, 12
    actions_symbols = {0: '^', 1: '>', 2: 'v', 3: '<'}
    
    print("\nLearned Greedy Policy (Arrows):")
    for r in range(height):
        row_str = []
        for c in range(width):
            state = r * width + c
            
            # Start is at state 36 (row 3, col 0)
            # Goal is at state 47 (row 3, col 11)
            # Cliff is states 37-46 (row 3, col 1-10)
            if state == 47:
                row_str.append("[ G ]")
            elif r == 3 and 1 <= c <= 10:
                row_str.append("[ C ]")
            elif state == 36:
                row_str.append("[ S ]")
            else:
                best_action = np.argmax(agent.Q[state])
                row_str.append(f"  {actions_symbols[best_action]}  ")
        print(" ".join(row_str))

if __name__ == "__main__":
    # Train the agent
    agent, history = train_agent(episodes=500, alpha=0.5, epsilon=0.1)
    
    # Print the resulting policy
    print_learned_policy(agent)
