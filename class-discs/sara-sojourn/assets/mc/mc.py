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

def generate_html_report(env, alpha, epsilon, mc_data, mc_viz):
    """Creates a unified HTML dashboard with all MC results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "mc_dashboard.html")
    
    # Format the DataFrames for HTML
    mc_html = mc_data.to_html(classes='table table-striped table-hover', float_format=lambda x: f"{x:.4f}")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Monte Carlo Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f4f7f6; padding: 20px; }}
            .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .viz-img {{ max-width: 100%; height: auto; border-radius: 8px; }}
            .table-container {{ max-height: 500px; overflow-y: auto; font-size: 0.85rem; }}
            h1, h2 {{ color: #1a5276; }}
            .concept-badge {{ font-size: 0.9rem; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="text-center mb-5">
                <h1>Monte Carlo: Learning from Experience</h1>
                <p class="lead">Exploring the FrozenLake when the Map is GONE</p>
                <div class="mt-3">
                    <span class="badge bg-danger concept-badge">Alpha: {alpha}</span>
                    <span class="badge bg-warning text-dark concept-badge">Epsilon: {epsilon}</span>
                    <span class="badge bg-secondary concept-badge">No Model Required</span>
                    <span class="badge bg-info concept-badge">Episode-by-Episode</span>
                </div>
            </div>

            <div class="row">
                <!-- MC Control Section -->
                <div class="col-lg-7">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h2 class="h5 mb-0">Monte Carlo Control (Epsilon-Greedy)</h2>
                        </div>
                        <div class="card-body text-center">
                            <img src="data:image/png;base64,{mc_viz}" class="viz-img mb-3" alt="MC Visualization">
                        </div>
                    </div>
                </div>

                <!-- Computation Trace Section -->
                <div class="col-lg-5">
                    <div class="card">
                        <div class="card-header bg-dark text-white">
                            <h2 class="h5 mb-0">Episode History Trace</h2>
                        </div>
                        <div class="card-body">
                            <div class="table-container">
                                {mc_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card mt-4">
                <div class="card-header bg-info text-white">
                    <h2 class="h5 mb-0">The Monte Carlo Story</h2>
                </div>
                <div class="card-body">
                    <ul>
                        <li><strong>No Map? No Problem:</strong> Unlike DP, Monte Carlo doesn't know the probabilities of the world. It simply walks and observes the <em>Return</em> ($G_t$).</li>
                        <li><strong>Ground Truth over Guesswork:</strong> MC does not <em>bootstrap</em>. It waits for the full episode to end before updating. It uses actual samples, not estimated next-state values.</li>
                        <li><strong>The Exploration Trade-off:</strong> Since we don't have a map, we use <em>$\epsilon$-greedy</em>. We follow our best guess most of the time, but sometimes we explore randomly to find hidden treasures (or holes!).</li>
                        <li><strong>High Variance:</strong> Because MC relies on full episodes, a single lucky (or unlucky) walk can significantly sway the value estimates until many more samples are collected.</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nMonte Carlo dashboard created at: {output_path}")

def get_mc_viz_base64(Q, env, title):
    """Generates a visualization for Q-values and returns it as base64."""
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))
    
    # Calculate V from Q (V is the max Q for each state)
    V = np.max(Q, axis=1)
    V_grid = V.reshape((grid_size, grid_size))
    
    # Best actions
    policy = np.argmax(Q, axis=1)
    policy_grid = policy.reshape((grid_size, grid_size))
    
    desc = env.unwrapped.desc
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(V_grid, annot=True, fmt=".3f", cmap="YlOrRd", cbar=True, square=True, ax=ax)
    
    dx = {0: -0.3, 1: 0, 2: 0.3, 3: 0}
    dy = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    
    for y in range(grid_size):
        for x in range(grid_size):
            char = desc[y][x].decode('utf-8')
            ax.text(x + 0.5, y + 0.2, char, weight='bold', size=12, ha='center', va='center', color='black')
            if char not in ['H', 'G']:
                action = policy_grid[y, x]
                ax.arrow(x + 0.5, y + 0.7, dx[action], dy[action], 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    plt.title(title)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    """Picks an action using epsilon-greedy strategy."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

def monte_carlo_control(env, episodes=5000, gamma=0.95, alpha=0.01, epsilon=0.1):
    """
    Monte Carlo Control (On-policy Epsilon-Greedy).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    episode_data = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []
        terminated = False
        truncated = False
        
        # 1. Generate an episode
        while not (terminated or truncated):
            action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # 2. Process the episode (First-visit MC)
        G = 0
        visited_sa = set()
        for i in reversed(range(len(episode))):
            s, a, r = episode[i]
            G = r + gamma * G
            
            if (s, a) not in visited_sa:
                # Update Q using Incremental Mean (Alpha-constant)
                Q[s, a] += alpha * (G - Q[s, a])
                visited_sa.add((s, a))
        
        if (ep + 1) % 100 == 0 or ep < 10:
            episode_data.append({
                "Episode": ep + 1,
                "Length": len(episode),
                "Total Reward": sum(r for s, a, r in episode),
                "V(Start)": np.max(Q[0])
            })
            
    return Q, pd.DataFrame(episode_data)

def main():
    # Use Slippery=True because MC is designed for stochasticity/unknown environments
    env = gym.make("FrozenLake-v1", is_slippery=True)
    gamma = 0.99
    alpha = 0.05
    epsilon = 0.2
    
    print("--- Running Monte Carlo Learning (Wait for episodes...) ---")
    Q, history_df = monte_carlo_control(env, episodes=2000, gamma=gamma, alpha=alpha, epsilon=epsilon)
    
    # Visualization
    mc_viz = get_mc_viz_base64(Q, env, "Monte Carlo Control: Final Optimal Q-Policy")
    
    # Generate Unified HTML Report
    generate_html_report(env, alpha, epsilon, history_df, mc_viz)

if __name__ == "__main__":
    main()
