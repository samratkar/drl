import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Blackjack Environment
STICK = 0
HIT = 1
ACTIONS = [STICK, HIT]

def play_blackjack(policy_func):
    """Simulates a game of Blackjack."""
    player_sum = np.random.randint(12, 22)
    dealer_card = np.random.randint(1, 11)
    usable_ace = np.random.choice([True, False])
    
    state = (player_sum, dealer_card, usable_ace)
    trajectory = []
    
    # Player's turn
    while True:
        action = policy_func(player_sum)
        trajectory.append((state, action))
        if action == STICK: break
        
        card = min(np.random.randint(1, 14), 10)
        player_sum += card
        if player_sum > 21:
            if usable_ace:
                player_sum -= 10
                usable_ace = False
            else:
                return trajectory, -1, state
        state = (player_sum, dealer_card, usable_ace)
        
    # Dealer's turn
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
                return trajectory, 1, state
    
    if player_sum > dealer_sum: return trajectory, 1, state
    elif player_sum < dealer_sum: return trajectory, -1, state
    else: return trajectory, 0, state

def target_policy(player_sum):
    return STICK if player_sum >= 20 else HIT

def behavior_policy(player_sum):
    return np.random.choice(ACTIONS)

def importance_sampling_blackjack(target_state, episodes=10000, runs=100):
    # Figure 5.3: Ordinary vs Weighted IS
    # Target state: Player 13, Dealer 2, Usable Ace True
    
    true_value = -0.27726 # Value for this state under target policy (approx)
    
    ord_is_errors = np.zeros((runs, episodes))
    wei_is_errors = np.zeros((runs, episodes))
    
    for r in tqdm(range(runs)):
        ord_sum = 0
        wei_num = 0
        wei_den = 0
        
        for e in range(episodes):
            # Start episode from target state
            p_sum, d_card, u_ace = target_state
            state = (p_sum, d_card, u_ace)
            trajectory = []
            
            # Behavior policy rollout
            curr_p_sum = p_sum
            curr_u_ace = u_ace
            while True:
                action = behavior_policy(curr_p_sum)
                trajectory.append(((curr_p_sum, d_card, curr_u_ace), action))
                if action == STICK: break
                card = min(np.random.randint(1, 14), 10)
                curr_p_sum += card
                if curr_p_sum > 21:
                    if curr_u_ace:
                        curr_p_sum -= 10
                        curr_u_ace = False
                    else:
                        reward = -1
                        break
                else:
                    reward = None # continues
            
            if reward is None:
                # Dealer's turn
                dealer_sum = d_card
                dealer_usable_ace = (d_card == 1)
                if dealer_usable_ace: dealer_sum += 10
                while dealer_sum < 17:
                    card = min(np.random.randint(1, 14), 10)
                    dealer_sum += card
                    if dealer_sum > 21:
                        if dealer_usable_ace:
                            dealer_sum -= 10
                            dealer_usable_ace = False
                        else:
                            reward = 1
                            break
                if reward is None:
                    if curr_p_sum > dealer_sum: reward = 1
                    elif curr_p_sum < dealer_sum: reward = -1
                    else: reward = 0
            
            # Calculate importance ratio rho
            rho = 1.0
            for (p_s, d_c, u_a), a in trajectory:
                pi_a = 1.0 if target_policy(p_s) == a else 0.0
                b_a = 0.5
                rho *= (pi_a / b_a)
                if rho == 0: break
            
            # Ordinary IS
            ord_sum += rho * reward
            v_ord = ord_sum / (e + 1)
            ord_is_errors[r, e] = (v_ord - true_value)**2
            
            # Weighted IS
            wei_num += rho * reward
            wei_den += rho
            v_wei = wei_num / wei_den if wei_den != 0 else 0
            wei_is_errors[r, e] = (v_wei - true_value)**2
            
    return np.mean(ord_is_errors, axis=0), np.mean(wei_is_errors, axis=0)

if __name__ == "__main__":
    target_state = (13, 2, True)
    ord_err, wei_err = importance_sampling_blackjack(target_state, episodes=10000, runs=100)
    
    plt.plot(ord_err, label='Ordinary Importance Sampling')
    plt.plot(wei_err, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('MS Error (log scale)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Figure 5.3: Ordinary vs Weighted IS on Blackjack')
    plt.legend()
    plt.show()
