import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import base64
from io import BytesIO

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_viz_base64(Q, env, title):
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))
    V = np.max(Q, axis=1).reshape((grid_size, grid_size))
    policy = np.argmax(Q, axis=1).reshape((grid_size, grid_size))
    
    desc = env.unwrapped.desc
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(V, annot=True, fmt=".3f", cmap="viridis", cbar=False, square=True, ax=ax)
    
    dx = {0: -0.3, 1: 0, 2: 0.3, 3: 0}
    dy = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    
    for y in range(grid_size):
        for x in range(grid_size):
            char = desc[y][x].decode('utf-8')
            ax.text(x + 0.5, y + 0.2, char, weight='bold', size=10, ha='center', va='center', color='black')
            if char not in ['H', 'G']:
                a = policy[y, x]
                ax.arrow(x + 0.5, y + 0.7, dx[a], dy[a], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    plt.title(title)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def generate_html_report(env, alpha, gamma, epsilon, q_data, sarsa_data, q_viz, sarsa_viz):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "td_dashboard.html")
    
    q_html = q_data.to_html(classes='table table-sm table-striped', float_format=lambda x: f"{x:.4f}", index=False)
    sarsa_html = sarsa_data.to_html(classes='table table-sm table-striped', float_format=lambda x: f"{x:.4f}", index=False)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TD Learning Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f8f9fa; padding: 20px; }}
            .card {{ margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .viz-img {{ max-width: 100%; border-radius: 4px; }}
            .scroll-table {{ max-height: 300px; overflow-y: auto; font-size: 0.85rem; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="text-center mb-4">
                <h2>Temporal Difference: Step-by-Step Learning</h2>
                <span class="badge bg-primary">Alpha: {alpha}</span>
                <span class="badge bg-success">Gamma: {gamma}</span>
                <span class="badge bg-warning text-dark">Epsilon: {epsilon}</span>
                <span class="badge bg-secondary">Slippery Map</span>
            </div>

            <div class="row">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header bg-dark text-white">Q-Learning (Off-Policy / Optimistic)</div>
                        <div class="card-body text-center">
                            <img src="data:image/png;base64,{q_viz}" class="viz-img mb-3">
                            <div class="scroll-table">{q_html}</div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header bg-dark text-white">SARSA (On-Policy / Cautious)</div>
                        <div class="card-body text-center">
                            <img src="data:image/png;base64,{sarsa_viz}" class="viz-img mb-3">
                            <div class="scroll-table">{sarsa_html}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"TD Dashboard generated at: {output_path}")

def epsilon_greedy(Q, s, epsilon, n_actions):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[s])

def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    data = []
    for ep in range(episodes):
        s, _ = env.reset()
        total_r = 0
        steps = 0
        while True:
            a = epsilon_greedy(Q, s, epsilon, env.action_space.n)
            s_prime, r, terminated, truncated, _ = env.step(a)
            td_target = r + gamma * np.max(Q[s_prime]) * (not terminated)
            Q[s, a] += alpha * (td_target - Q[s, a])
            total_r += r
            steps += 1
            s = s_prime
            if terminated or truncated: break
        if ep % 500 == 0 or ep == episodes - 1:
            data.append({"Episode": ep, "Steps": steps, "Reward": total_r, "V(Start)": np.max(Q[0])})
    return Q, pd.DataFrame(data)

def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    data = []
    for ep in range(episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, epsilon, env.action_space.n)
        total_r = 0
        steps = 0
        while True:
            s_prime, r, terminated, truncated, _ = env.step(a)
            a_prime = epsilon_greedy(Q, s_prime, epsilon, env.action_space.n)
            td_target = r + gamma * Q[s_prime, a_prime] * (not terminated)
            Q[s, a] += alpha * (td_target - Q[s, a])
            total_r += r
            steps += 1
            s, a = s_prime, a_prime
            if terminated or truncated: break
        if ep % 500 == 0 or ep == episodes - 1:
            data.append({"Episode": ep, "Steps": steps, "Reward": total_r, "V(Start)": np.max(Q[0])})
    return Q, pd.DataFrame(data)

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    episodes = 5000
    
    print("--- Running Q-Learning ---")
    q_Q, q_data = q_learning(env, episodes, alpha, gamma, epsilon)
    q_viz = get_viz_base64(q_Q, env, "Q-Learning (Optimistic)")
    
    print("--- Running SARSA ---")
    s_Q, s_data = sarsa(env, episodes, alpha, gamma, epsilon)
    s_viz = get_viz_base64(s_Q, env, "SARSA (Cautious)")
    
    generate_html_report(env, alpha, gamma, epsilon, q_data, s_data, q_viz, s_viz)

if __name__ == "__main__":
    main()
