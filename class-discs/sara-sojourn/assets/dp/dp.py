import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import base64
from io import BytesIO

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_html_report(env, gamma, pi_data, vi_data, pi_viz, vi_viz):
    """Creates a unified HTML dashboard with all DP results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "dp_dashboard.html")
    
    # Format the DataFrames for HTML
    pi_html = pi_data.to_html(classes='table table-striped table-hover', float_format=lambda x: f"{x:.4f}")
    vi_html = vi_data.to_html(classes='table table-striped table-hover', float_format=lambda x: f"{x:.4f}")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dynamic Programming Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f8f9fa; padding: 20px; }}
            .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .viz-img {{ max-width: 100%; height: auto; border-radius: 8px; }}
            .table-container {{ max-height: 400px; overflow-y: auto; font-size: 0.85rem; }}
            h1, h2 {{ color: #2c3e50; }}
            .concept-badge {{ font-size: 0.9rem; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="text-center mb-5">
                <h1>Dynamic Programming: The FrozenLake Journey</h1>
                <p class="lead">Exploring Convergence, Value Functions, and Optimal Policies</p>
                <div class="mt-3">
                    <span class="badge bg-primary concept-badge">Gamma: {gamma}</span>
                    <span class="badge bg-success concept-badge">Deterministic Map</span>
                    <span class="badge bg-info concept-badge">Bellman Optimality</span>
                </div>
            </div>

            <div class="row">
                <!-- Policy Iteration Section -->
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header bg-dark text-white">
                            <h2 class="h5 mb-0">1. Policy Iteration (The Ladder)</h2>
                        </div>
                        <div class="card-body text-center">
                            <img src="data:image/png;base64,{pi_viz}" class="viz-img mb-3" alt="PI Visualization">
                            <div class="table-container">
                                <h6>Computation Trace (Value Evaluation)</h6>
                                {pi_html}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Value Iteration Section -->
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header bg-dark text-white">
                            <h2 class="h5 mb-0">2. Value Iteration (The Shortcut)</h2>
                        </div>
                        <div class="card-body text-center">
                            <img src="data:image/png;base64,{vi_viz}" class="viz-img mb-3" alt="VI Visualization">
                            <div class="table-container">
                                <h6>Computation Trace (Max-Sweep)</h6>
                                {vi_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card mt-4">
                <div class="card-header bg-secondary text-white">
                    <h2 class="h5 mb-0">Pedagogical Insights</h2>
                </div>
                <div class="card-body">
                    <ul>
                        <li><strong>Convergence:</strong> Notice how the values in the tables stabilize. The change between iterations (Delta) eventually hits the threshold.</li>
                        <li><strong>Goal Propagation:</strong> Values are highest near the Goal (G) and decrease as they move towards the Start (S) due to the discount factor $\gamma$.</li>
                        <li><strong>Policy Arrows:</strong> These represent the 'Greedy' action based on the 'Perfect Map' ($env.P$).</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nUnified HTML dashboard created at: {output_path}")

def get_viz_base64(V, policy, env, title):
    """Generates a visualization and returns it as a base64 string."""
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))
    V_grid = V.reshape((grid_size, grid_size))
    policy_grid = policy.reshape((grid_size, grid_size))
    desc = env.unwrapped.desc
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(V_grid, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, square=True, ax=ax)
    
    dx = {0: -0.3, 1: 0, 2: 0.3, 3: 0}
    dy = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    
    for y in range(grid_size):
        for x in range(grid_size):
            char = desc[y][x].decode('utf-8')
            ax.text(x + 0.5, y + 0.2, char, weight='bold', size=10, ha='center', va='center', color='black')
            if char not in ['H', 'G']:
                action = policy_grid[y, x]
                ax.arrow(x + 0.5, y + 0.7, dx[action], dy[action], 
                         head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    plt.title(title)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def extract_policy(V, env, gamma):
    """Extracts the greedy policy from a given value function."""
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        q_values = [sum(prob * (reward + gamma * V[next_s]) 
                      for prob, next_s, reward, terminated in env.unwrapped.P[s][a])
                   for a in range(env.action_space.n)]
        policy[s] = np.argmax(q_values)
    return policy

def policy_evaluation(policy, env, gamma, theta=1e-8):
    """Calculates the true value of a specific policy."""
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v_old = V[s]
            a = policy[s]
            V[s] = sum(prob * (reward + gamma * V[next_s]) 
                      for prob, next_s, reward, terminated in env.unwrapped.P[s][a])
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta: break
    return V

def policy_iteration(env, gamma=0.99):
    """Policy Iteration: The Ladder Climbing Strategy."""
    policy = np.random.randint(0, env.action_space.n, size=env.observation_space.n)
    while True:
        V = policy_evaluation(policy, env, gamma)
        policy_stable = True
        for s in range(env.observation_space.n):
            old_action = policy[s]
            q_values = [sum(prob * (reward + gamma * V[next_s]) 
                          for prob, next_s, reward, terminated in env.unwrapped.P[s][a])
                       for a in range(env.action_space.n)]
            new_action = np.argmax(q_values)
            policy[s] = new_action
            if old_action != new_action: policy_stable = False
        if policy_stable: break
    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    """Value Iteration: The Shortcut Strategy."""
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v_old = V[s]
            q_values = [sum(prob * (reward + gamma * V[next_s]) 
                          for prob, next_s, reward, terminated in env.unwrapped.P[s][a])
                       for a in range(env.action_space.n)]
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta: break
    policy = extract_policy(V, env, gamma)
    return policy, V

def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    gamma = 0.95
    
    print("--- Running Dynamic Programming Algorithms ---")
    
    # Trace for PI (Evaluation iterations of the FINAL policy for demonstration)
    pi_policy, pi_v = policy_iteration(env, gamma)
    v_trace = np.zeros(env.observation_space.n)
    pi_history = []
    for _ in range(20):
        pi_history.append(v_trace.copy())
        v_next = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            a = pi_policy[s]
            v_next[s] = sum(prob * (reward + gamma * v_trace[next_s]) 
                           for prob, next_s, reward, terminated in env.unwrapped.P[s][a])
        v_trace = v_next
    df_pi = pd.DataFrame(pi_history, columns=[f"S{i}" for i in range(env.observation_space.n)])

    # Trace for VI
    v_vi = np.zeros(env.observation_space.n)
    vi_history = []
    for _ in range(20):
        vi_history.append(v_vi.copy())
        v_next = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            q_sa = [sum(prob * (reward + gamma * v_vi[next_s]) 
                       for prob, next_s, reward, terminated in env.unwrapped.P[s][a])
                   for a in range(env.action_space.n)]
            v_next[s] = max(q_sa)
        v_vi = v_next
    df_vi = pd.DataFrame(vi_history, columns=[f"S{i}" for i in range(env.observation_space.n)])

    # Visualizations
    pi_viz = get_viz_base64(pi_v, pi_policy, env, "Policy Iteration Final State")
    vi_policy, vi_v = value_iteration(env, gamma)
    vi_viz = get_viz_base64(vi_v, vi_policy, env, "Value Iteration Final State")
    
    # Generate Report
    generate_html_report(env, gamma, df_pi, df_vi, pi_viz, vi_viz)

if __name__ == "__main__":
    main()
