import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Blackjack Environment constants
# Actions
STICK = 0
HIT = 1
ACTIONS = [STICK, HIT]

# Policy for prediction: stick on 20 or 21, hit otherwise
def target_policy_prediction(player_sum):
    return STICK if player_sum >= 20 else HIT

def play_blackjack(policy_func=None, initial_state=None, initial_action=None):
    """
    Simulates a game of Blackjack.
    If policy_func is None, it uses the provided initial_action then random actions (for Exploring Starts).
    """
    # Initialize state
    if initial_state:
        player_sum, dealer_card, usable_ace = initial_state
    else:
        # Player sum: 12-21
        player_sum = np.random.randint(12, 22)
        # Dealer's showing card: 1-10
        dealer_card = np.random.randint(1, 11)
        # Usable ace: 0 or 1
        usable_ace = np.random.choice([True, False])

    state = (player_sum, dealer_card, usable_ace)
    trajectory = []

    # Player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        elif policy_func:
            action = policy_func(player_sum)
        else:
            action = np.random.choice(ACTIONS)

        trajectory.append((state, action))

        if action == STICK:
            break
        
        # HIT
        card = min(np.random.randint(1, 14), 10) # 1-10, face cards are 10
        player_sum += card
        if player_sum > 21:
            if usable_ace:
                player_sum -= 10
                usable_ace = False
            else:
                return trajectory, -1 # Player busts

        state = (player_sum, dealer_card, usable_ace)

    # Dealer's turn (fixed policy: hit until >= 17)
    dealer_sum = dealer_card
    dealer_usable_ace = (dealer_card == 1)
    if dealer_usable_ace: dealer_sum += 10

    while dealer_sum < 17:
        card = min(np.random.randint(1, 14), 10)
        dealer_sum += card
        if dealer_sum > 21:
            if dealer_usable_ace:
                dealer_sum -= 10
                dealer_usable_ace = False
            else:
                return trajectory, 1 # Dealer busts

    # Compare sums
    if player_sum > dealer_sum:
        return trajectory, 1
    elif player_sum < dealer_sum:
        return trajectory, -1
    else:
        return trajectory, 0

def mc_prediction(episodes=10000):
    V = np.zeros((10, 10, 2)) # player_sum (12-21), dealer_card (1-10), usable_ace (0,1)
    returns_sum = np.zeros_like(V)
    returns_count = np.zeros_like(V)

    for _ in range(episodes):
        trajectory, reward = play_blackjack(policy_func=target_policy_prediction)
        for (p_sum, d_card, u_ace), action in trajectory:
            if p_sum >= 12:
                idx = (p_sum - 12, d_card - 1, int(u_ace))
                returns_sum[idx] += reward
                returns_count[idx] += 1
                V[idx] = returns_sum[idx] / returns_count[idx]
    return V

def mc_es_control(episodes=500000):
    # Q(s, a)
    Q = np.zeros((10, 10, 2, 2)) # player_sum, dealer_card, usable_ace, action
    returns_sum = np.zeros_like(Q)
    returns_count = np.zeros_like(Q)
    policy = np.zeros((10, 10, 2), dtype=int)
    # Initial policy: STICK on 20, 21
    for p in range(10):
        policy[p, :, :] = STICK if (p + 12) >= 20 else HIT

    for _ in tqdm(range(episodes)):
        # Exploring Starts
        p_sum = np.random.randint(12, 22)
        d_card = np.random.randint(1, 11)
        u_ace = np.random.choice([True, False])
        action = np.random.choice(ACTIONS)
        
        trajectory, reward = play_blackjack(initial_state=(p_sum, d_card, u_ace), initial_action=action)
        
        visited = set()
        for (p_s, d_c, u_a), act in trajectory:
            if p_s >= 12:
                state_action = (p_s - 12, d_c - 1, int(u_a), act)
                if state_action not in visited:
                    returns_sum[state_action] += reward
                    returns_count[state_action] += 1
                    Q[state_action] = returns_sum[state_action] / returns_count[state_action]
                    
                    # Update policy greedily
                    s_idx = (p_s - 12, d_c - 1, int(u_a))
                    policy[s_idx] = np.argmax(Q[s_idx])
                    visited.add(state_action)
    return policy, Q

if __name__ == "__main__":
    print("Running MC Prediction...")
    V = mc_prediction(50000)
    print("MC Prediction Finished.")
    
    print("Running MC ES Control...")
    policy, Q = mc_es_control(100000)
    print("MC ES Control Finished.")
    
    # Simple policy visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(policy[:, :, 1], origin='lower', extent=[1, 10, 12, 21])
    ax[0].set_title("Optimal Policy (Usable Ace)")
    ax[0].set_xlabel("Dealer Card")
    ax[0].set_ylabel("Player Sum")
    
    ax[1].imshow(policy[:, :, 0], origin='lower', extent=[1, 10, 12, 21])
    ax[1].set_title("Optimal Policy (No Usable Ace)")
    ax[1].set_xlabel("Dealer Card")
    ax[1].set_ylabel("Player Sum")
    
    plt.tight_layout()
    plt.show()
