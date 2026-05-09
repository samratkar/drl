"""
Handling Large Action Spaces: Sampling-based approach for continuous spaces.
"""
import numpy as np

class MockActionSpace:
    def sample(self):
        return np.random.uniform(-1, 1)

def sample_based_best_action(s, V, P, gamma, action_space, num_samples=100):
    """
    For continuous actions, sample a subset and pick the best one.
    """
    sampled_actions = [action_space.sample() for _ in range(num_samples)]
    q_values = []
    for a in sampled_actions:
        # Assuming we can query the model for these sampled actions
        q_sa = sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in P[(s, a)])
        q_values.append(q_sa)
    return sampled_actions[np.argmax(q_values)]

if __name__ == "__main__":
    action_space = MockActionSpace()
    V = {0: 0, 1: 10}
    
    # Mocking the P dictionary for any sampled actions
    class MockDict(dict):
        def __getitem__(self, key):
            s, a = key
            return [(1.0, 1, a * 10)] # Higher reward for positive 'a'
    
    best_a = sample_based_best_action(0, V, MockDict(), 0.9, action_space, num_samples=20)
    print(f"Best sampled action: {best_a}")
