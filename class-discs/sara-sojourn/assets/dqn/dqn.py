import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import base64
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

def train_dqn(env, episodes=200):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    buffer = ReplayBuffer(10000)
    
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    target_update = 10
    
    history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(state)).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if len(buffer) > batch_size:
                s, a, r, s_prime, d = buffer.sample(batch_size)
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a).unsqueeze(1)
                r = torch.FloatTensor(r).unsqueeze(1)
                s_prime = torch.FloatTensor(s_prime)
                d = torch.FloatTensor(d).unsqueeze(1)
                
                q_values = q_net(s).gather(1, a)
                with torch.no_grad():
                    max_next_q = target_net(s_prime).max(1)[0].unsqueeze(1)
                    target_q = r + gamma * max_next_q * (1 - d)
                
                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if ep % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())
            
        history.append({"Episode": ep, "Total Reward": total_reward, "Epsilon": epsilon})
        
    return pd.DataFrame(history)

def generate_html_report(history_df, reward_viz):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "dqn_dashboard.html")
    
    hist_html = history_df.tail(20).to_html(classes='table table-sm table-striped', float_format=lambda x: f"{x:.2f}", index=False)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deep Q-Network Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f1f4f8; padding: 20px; }}
            .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .viz-img {{ max-width: 100%; border-radius: 8px; }}
            .scroll-table {{ max-height: 400px; overflow-y: auto; font-size: 0.85rem; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="text-center mb-4">
                <h2>Deep Q-Network (DQN): The Neural Brain</h2>
                <p class="lead">Training on continuous features (CartPole) with Experience Replay & Target Networks</p>
            </div>

            <div class="row">
                <div class="col-lg-8">
                    <div class="card">
                        <div class="card-header bg-primary text-white">Learning Curve (Total Reward over Episodes)</div>
                        <div class="card-body text-center">
                            <img src="data:image/png;base64,{reward_viz}" class="viz-img mb-3">
                        </div>
                    </div>
                </div>

                <div class="col-lg-4">
                    <div class="card">
                        <div class="card-header bg-dark text-white">Latest Episodes Log</div>
                        <div class="card-body text-center">
                            <div class="scroll-table">{hist_html}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card mt-2">
                <div class="card-header bg-info text-white">Why DQN is Necessary</div>
                <div class="card-body">
                    <ul>
                        <li><strong>Continuous Spaces:</strong> CartPole has infinite possible states (angles, velocities). A Q-table cannot hold infinite rows. A Neural Net approximates the value.</li>
                        <li><strong>Experience Replay:</strong> We save steps in a buffer and train on random samples. This prevents the Neural Net from overfitting to correlated sequences of states.</li>
                        <li><strong>Target Network:</strong> We use an older, frozen copy of the network to calculate the TD Target. This stops the "dog chasing its tail" problem of a constantly moving target.</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"DQN Dashboard generated at: {output_path}")

def main():
    env = gym.make("CartPole-v1")
    print("--- Training basic DQN on CartPole-v1 (approx 200 episodes) ---")
    
    history_df = train_dqn(env, episodes=200)
    
    # Plotting the reward curve
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=history_df, x="Episode", y="Total Reward", ax=ax, color='blue', alpha=0.6)
    # Add a rolling average
    history_df['Rolling Reward'] = history_df['Total Reward'].rolling(10).mean()
    sns.lineplot(data=history_df, x="Episode", y="Rolling Reward", ax=ax, color='red', linewidth=2)
    plt.title("DQN Learning Progress")
    plt.ylabel("Reward (Survival Time)")
    
    reward_viz = fig_to_base64(fig)
    plt.close(fig)
    
    generate_html_report(history_df, reward_viz)

if __name__ == "__main__":
    main()
