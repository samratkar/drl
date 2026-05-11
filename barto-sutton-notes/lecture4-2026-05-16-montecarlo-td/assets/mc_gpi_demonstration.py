import numpy as np
import random
from collections import defaultdict

class SimpleGridWorld:
    """
    A simple 3x3 GridWorld for pedagogical demonstration.
    0 1 2
    3 4 5
    6 7 8 (8 is Goal, Reward +10; all other steps -1)
    """
    def __init__(self):
        self.size = 3
        self.goal = 8
        self.reset()

    def reset(self, start_state=None):
        if start_state is not None:
            self.state = start_state
        else:
            # Exploring Starts: start at any non-terminal state
            self.state = random.choice(range(self.goal))
        return self.state

    def step(self, action):
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        row, col = divmod(self.state, self.size)
        if action == 0: row = max(0, row - 1)
        elif action == 1: row = min(self.size - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(self.size - 1, col + 1)
        
        self.state = row * self.size + col
        
        if self.state == self.goal:
            return self.state, 10, True
        else:
            return self.state, -1, False

def demonstrate_mc_gpi():
    print("=== Monte Carlo GPI Demonstration (Exploring Starts) ===")
    print("Environment: 3x3 Grid. Goal is State 8. Reward +10 at Goal, -1 otherwise.\n")

    env = SimpleGridWorld()
    gamma = 0.9
    
    # Initialize Q(s,a) and Returns(s,a)
    Q = defaultdict(lambda: np.zeros(4))
    returns_sum = defaultdict(lambda: np.zeros(4))
    returns_count = defaultdict(lambda: np.zeros(4))
    
    # Initialize Policy (arbitrary, let's say all Right)
    policy = defaultdict(lambda: 3)

    num_iterations = 5 # Small number for demonstration
    
    for i in range(num_iterations):
        print(f"--- GPI Iteration {i+1} ---")
        
        # 1. EVALUATION PHASE: Generate an episode with Exploring Starts
        # Sutton & Barto 5.3: "The first step of each episode is a state-action pair..."
        start_state = random.choice(range(8))
        start_action = random.choice(range(4))
        
        print(f"[Evaluation] Starting episode at State {start_state} with Action {start_action}")
        
        episode = []
        state = env.reset(start_state)
        action = start_action
        
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if not done:
                action = policy[state] # Follow current policy for the rest of the episode
        
        print(f"[Evaluation] Episode path: {[(s, a) for s, a, r in episode]} -> Goal")
        
        # 2. UPDATE PHASE: Calculate returns and update Q
        G = 0
        visited_sa = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            
            # First-visit check for state-action pair
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]
                print(f"   Update Q({s}, {a}) to {Q[s][a]:.2f} (Return G={G:.2f})")

        # 3. IMPROVEMENT PHASE: Make policy greedy w.r.t Q
        old_policy = policy.copy()
        for s in range(8):
            policy[s] = np.argmax(Q[s])
        
        # Check if policy changed
        changes = [s for s in range(8) if old_policy[s] != policy[s]]
        if changes:
            print(f"[Improvement] Policy updated for states: {changes}")
        else:
            print("[Improvement] Policy stable (no changes)")
        
        print(f"Current Policy (0:U, 1:D, 2:L, 3:R): {[policy[s] for s in range(8)]}\n")

    print("=== Conclusion ===")
    print("Notice how GPI works in MC:")
    print("1. Evaluation: We don't have a model, so we *wait* for the episode to finish.")
    print("2. Evaluation: We calculate the actual Return (G) from each state-action pair.")
    print("3. Improvement: We immediately update our policy to take the action with the highest estimated Return.")
    print("The 'Exploring Starts' ensure we eventually try all actions from all states.")

if __name__ == "__main__":
    demonstrate_mc_gpi()
