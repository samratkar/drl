import numpy as np
from enum import Enum
from scipy.stats import poisson

# 1. Dynamics Representation: p(s', r | s, a) in a dictionary.
# Representing as {(state, action): [(probability, next_state, reward), ...]}
dynamics = {
    (0, "search"): [(0.7, 0, 1), (0.3, 1, 1)],
    (0, "wait"): [(1.0, 0, 0.5)],
    (1, "search"): [(0.8, 1, 1), (0.2, 0, -3)],
    (1, "wait"): [(1.0, 1, 0.5)],
    (1, "recharge"): [(1.0, 0, 0)]
}

# 2. Policy Definition: Function get_action(policy, state) for deterministic and stochastic policies.
def get_action(policy, state):
    """
    Handles both deterministic: {s: a} and stochastic: {s: {a: prob}} policies.
    """
    action_info = policy[state]
    if isinstance(action_info, dict):
        # Stochastic policy
        actions = list(action_info.keys())
        probs = list(action_info.values())
        return np.random.choice(actions, p=probs)
    # Deterministic policy
    return action_info

# 3. In-place Update: Code for in-place update of V[s].
# This updates the value function directly without keeping a copy of the old values.
def in_place_update(s, new_value, V):
    V[s] = new_value

# 4. Termination Condition: while loop for Policy Evaluation with theta.
def policy_evaluation(policy, dynamics, states, gamma, theta):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # simplified bellman update for a specific policy
            new_v = sum(p * (r + gamma * V[s_prime]) 
                        for p, s_prime, r in dynamics[(s, policy[s])])
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
    return V

# 5. Policy Improvement Logic: Logic to find the greedy action for state s.
def get_greedy_action(s, V, actions, dynamics, gamma):
    action_values = []
    for a in actions:
        q_sa = sum(p * (r + gamma * V[s_prime]) 
                   for p, s_prime, r in dynamics[(s, a)])
        action_values.append(q_sa)
    return actions[np.argmax(action_values)]

# 6. Value Iteration Core: Single line of code for the V[s] update in Value Iteration.
# V[s] = max(sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in dynamics[(s, a)]) for a in actions)

# 7. Jack's Car Rental (Poisson): Pre-compute Poisson probabilities up to max_cars.
def precompute_poisson(max_cars, lam):
    # Returns a list where index k is P(X=k)
    return [poisson.pmf(k, lam) for k in range(max_cars + 1)]

# 8. Gambler's Problem Stakes: Function get_possible_stakes(s, goal).
def get_possible_stakes(s, goal):
    # Stake must be at least 0 and at most the amount you have (s) or need (goal - s)
    return range(min(s, goal - s) + 1)

# 9. Tie-breaking: argmax that handles multiple max actions.
def stable_argmax(values):
    """
    Returns a random choice among the indices that have the maximum value.
    """
    max_val = np.max(values)
    indices = np.where(np.array(values) == max_val)[0]
    return np.random.choice(indices)

# 10. State Mapping: Lambda to convert (r, c) to 1D index i.
get_1d_index = lambda r, c, num_cols: r * num_cols + c

# 11. Absorbing State Implementation: Representing terminal states in dynamics.
# A terminal state 'T' stays in 'T' with 0 reward for any action.
def add_absorbing_state(dynamics, terminal_state, actions):
    for a in actions:
        dynamics[(terminal_state, a)] = [(1.0, terminal_state, 0.0)]

# 12. Asynchronous DP (Random): Snippet to update a random state.
def async_update_random_state(states, V, actions, dynamics, gamma):
    s = np.random.choice(states)
    # Perform a single value iteration update on state s
    V[s] = max(sum(p * (r + gamma * V[s_prime]) 
                   for p, s_prime, r in dynamics[(s, a)]) 
               for a in actions)

# 13. Model-Based vs. Model-Free: Explain/show difference in input types.
"""
Model-Based:
Input: The full environment model p(s', r | s, a). 
Code usually iterates over ALL possible outcomes.
"""
def model_based_update(s, a, model, V, gamma):
    # 'model' is p(s', r | s, a)
    return sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in model[(s, a)])

"""
Model-Free:
Input: Samples (experience) from the environment: (s, a, r, s').
Code updates using a single sample and a learning rate.
"""
def model_free_update(s, a, r, s_prime, V, gamma, alpha):
    # No model, just the transition experience
    target = r + gamma * V[s_prime]
    V[s] += alpha * (target - V[s])

# 14. Efficiency in Summation: Optimize summation for sparse transitions.
def efficient_sum(s, a, sparse_dynamics, V, gamma):
    # sparse_dynamics[(s, a)] only contains entries where p > 0
    return sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in sparse_dynamics[(s, a)])

# 15. Discount Factor Handling: calculate_q_value(s, a, V, P, gamma).
def calculate_q_value(s, a, V, P, gamma):
    """
    Calculates Q(s, a) using the transition model P and value function V.
    """
    return sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in P[(s, a)])

# 16. Verifying Convergence: Function is_optimal(V, P, gamma, theta).
def is_optimal(V, states, actions, P, gamma, theta):
    """
    Checks if V satisfies the Bellman Optimality Equation within tolerance theta.
    """
    for s in states:
        max_q = max(sum(p * (r + gamma * V[s_prime]) for p, s_prime, r in P[(s, a)]) 
                    for a in actions)
        if abs(V[s] - max_q) > theta:
            return False
    return True

# 17. Handling Large Action Spaces: Sampling-based approach for continuous spaces.
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

# 18. Recycling Robot States: Enum class for states.
class RobotState(Enum):
    HIGH = 0
    LOW = 1

# 19. Policy Stability Check: Function are_policies_equal(pi1, pi2).
def are_policies_equal(pi1, pi2, states):
    """
    Returns True if policies provide the same action for all states.
    """
    return all(pi1[s] == pi2[s] for s in states)

# 20. Visualizing the Value Function: Reshape and print 1D array as 4x4 matrix.
def visualize_v_4x4(V_array):
    """
    Assumes V_array is a flat numpy array or list of 16 elements.
    """
    V_matrix = np.array(V_array).reshape(4, 4)
    print("Value Function Visualization (4x4):")
    print(np.round(V_matrix, 2))

if __name__ == "__main__":
    # Example visualization call
    V_dummy = np.random.rand(16)
    visualize_v_4x4(V_dummy)
