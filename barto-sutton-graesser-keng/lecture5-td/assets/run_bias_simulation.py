import numpy as np
import matplotlib.pyplot as plt

def run_simulation(runs=10000, episodes=300, epsilon=0.1, alpha=0.1, gamma=1.0):
    left_actions_q = np.zeros(episodes)
    left_actions_double_q = np.zeros(episodes)
    
    # Q-learning: Q-tables shape (runs, num_actions)
    Q_A = np.zeros((runs, 2))  # 0: Left, 1: Right
    Q_B = np.zeros((runs, 10)) # 10 actions from B
    
    for ep in range(episodes):
        explore_A = np.random.rand(runs) < epsilon
        greedy_A = np.argmax(Q_A, axis=1)
        random_A = np.random.randint(0, 2, size=runs)
        actions_A = np.where(explore_A, random_A, greedy_A)
        
        left_actions_q[ep] = np.mean(actions_A == 0)
        
        idx_right = np.where(actions_A == 1)[0]
        if len(idx_right) > 0:
            Q_A[idx_right, 1] += alpha * (0.0 - Q_A[idx_right, 1])
            
        idx_left = np.where(actions_A == 0)[0]
        if len(idx_left) > 0:
            max_Q_B = np.max(Q_B[idx_left], axis=1)
            Q_A[idx_left, 0] += alpha * (0.0 + gamma * max_Q_B - Q_A[idx_left, 0])
            
            explore_B = np.random.rand(len(idx_left)) < epsilon
            greedy_B = np.argmax(Q_B[idx_left], axis=1)
            random_B = np.random.randint(0, 10, size=len(idx_left))
            actions_B = np.where(explore_B, random_B, greedy_B)
            
            rewards_B = np.random.normal(-0.1, 1.0, size=len(idx_left))
            
            for i, agent_idx in enumerate(idx_left):
                act_b = actions_B[i]
                r = rewards_B[i]
                Q_B[agent_idx, act_b] += alpha * (r - Q_B[agent_idx, act_b])

    # Double Q-learning
    Q1_A = np.zeros((runs, 2))
    Q2_A = np.zeros((runs, 2))
    Q1_B = np.zeros((runs, 10))
    Q2_B = np.zeros((runs, 10))
    
    for ep in range(episodes):
        Q_sum_A = Q1_A + Q2_A
        explore_A = np.random.rand(runs) < epsilon
        greedy_A = np.argmax(Q_sum_A, axis=1)
        random_A = np.random.randint(0, 2, size=runs)
        actions_A = np.where(explore_A, random_A, greedy_A)
        
        left_actions_double_q[ep] = np.mean(actions_A == 0)
        
        update_Q1 = np.random.rand(runs) < 0.5
        
        idx_right = np.where(actions_A == 1)[0]
        for idx in idx_right:
            if update_Q1[idx]:
                Q1_A[idx, 1] += alpha * (0.0 - Q1_A[idx, 1])
            else:
                Q2_A[idx, 1] += alpha * (0.0 - Q2_A[idx, 1])
                
        idx_left = np.where(actions_A == 0)[0]
        for idx in idx_left:
            if update_Q1[idx]:
                best_act_B = np.argmax(Q1_B[idx])
                Q1_A[idx, 0] += alpha * (0.0 + gamma * Q2_B[idx, best_act_B] - Q1_A[idx, 0])
            else:
                best_act_B = np.argmax(Q2_B[idx])
                Q2_A[idx, 0] += alpha * (0.0 + gamma * Q1_B[idx, best_act_B] - Q2_A[idx, 0])
                
        if len(idx_left) > 0:
            Q_sum_B = Q1_B[idx_left] + Q2_B[idx_left]
            explore_B = np.random.rand(len(idx_left)) < epsilon
            greedy_B = np.argmax(Q_sum_B, axis=1)
            random_B = np.random.randint(0, 10, size=len(idx_left))
            actions_B = np.where(explore_B, random_B, greedy_B)
            
            rewards_B = np.random.normal(-0.1, 1.0, size=len(idx_left))
            
            for i, agent_idx in enumerate(idx_left):
                act_b = actions_B[i]
                r = rewards_B[i]
                if update_Q1[agent_idx]:
                    Q1_B[agent_idx, act_b] += alpha * (r - Q1_B[agent_idx, act_b])
                else:
                    Q2_B[agent_idx, act_b] += alpha * (r - Q2_B[agent_idx, act_b])
                    
    return left_actions_q, left_actions_double_q

def plot_results(left_q, left_dq, out_path):
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(left_q * 100, label='Q-learning', color='#E11D48', linewidth=2)
    plt.plot(left_dq * 100, label='Double Q-learning', color='#2563EB', linewidth=2)
    plt.axhline(y=5.0, color='#9CA3AF', linestyle='--', linewidth=1.5, label='Optimal (5% minimum)')
    
    plt.title('Comparison of Q-learning and Double Q-learning on Maximization Bias', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Episodes', fontsize=10, labelpad=8)
    plt.ylabel('% left actions from A', fontsize=10, labelpad=8)
    plt.xlim(0, 300)
    plt.ylim(0, 100)
    
    # Custom formatting to match the book
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
    plt.legend(frameon=True, facecolor='white', edgecolor='#E5E7EB', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6, color='#D1D5DB')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#9CA3AF')
    plt.gca().spines['bottom'].set_color('#9CA3AF')
    
    plt.tight_layout()
    plt.savefig(out_path, format='svg')
    plt.close()
    print("Graph saved successfully!")

if __name__ == "__main__":
    print("Running simulation...")
    left_q, left_dq = run_simulation(runs=10000, episodes=300)
    out_path = r"c:\github\drl\barto-sutton-notes\lecture5-td\assets\diagrams\maximization_bias.svg"
    plot_results(left_q, left_dq, out_path)
