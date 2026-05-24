# Deep Reinforcement Learning (DRL) Project

## Project Overview
This repository is an educational curriculum for Deep Reinforcement Learning. It implements algorithms ranging from foundational tabular methods (Dynamic Programming, Monte Carlo, TD Learning) to advanced deep reinforcement learning approaches (DQN, DDQN, PPO, Actor-Critic).

The project is built with **Python 3.12**, **PyTorch**, and **Gymnasium**.

## Project Structure
The repository is organized into sequential modules:
- **`1-introduction/`**: Core tabular RL concepts.
    - `dynamic-prog/`: Policy/Value iteration.
    - `montecarlo/`: Monte Carlo methods.
    - `assets/`: Custom environment definitions.
- **`3-sarsa/`**: Temporal Difference learning implementations (Sarsa, Q-Learning).
- **`cases/1-gym_atari_lunar_lander/`**: Progressive DQN implementation chain (Basic -> Replay Buffer -> Fixed Q -> DDQN -> Prioritized Replay).
- **`class-discs/BITS-DRL-S2-25_AIMLCZG512/lecture3-mdp-dp/assets/race-car/`**: Comprehensive RL comparison using the Race Car problem (DP, MC, TD, Q-Learning).
- **`class-discs/sara-sojourn/`**: Comparative dashboards and visualizations of different algorithms.
- **`books/memory/`**: Comprehensive notes on foundational texts like Sutton & Barto.
- **`assignments/`**: Course-related tasks and quizzes.

## Current Progress & Technical Context
### Race Car MDP Case Study
A specialized case study has been implemented at `.../assets/race-car/dp_race_car_demonstration.ipynb`.
- **States**: Cool, Warm, Overheated (Terminal).
- **Planning (DP)**: Implemented Policy Evaluation, Policy Iteration, and Value Iteration with full visibility into $\pi$ probabilities and $V$ estimates.
- **Learning (Model-Free)**: Implemented Monte Carlo, TD(0), and Q-Learning treating the transition model as a 'Black Box'.
- **Key Outcome**: The agent learns to drive FAST when Cool (higher reward, 0% risk) but SLOW when Warm (to avoid the -10 penalty from a 100% overheat risk).

### Implementation Logic
- **Policy Iteration**: Measure ($V_\pi$) -> Improve ($Q_\pi$ scorecard + `argmax`).
- **Value Iteration**: Greedy shortcut using `max()` directly on the Bellman Optimality Equation.
- **Q-Learning**: 'Actual' environment interaction using sample updates and a learning rate ($\alpha$) to average out noise without a model.

## Development Setup
### Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`

### Installation
```bash
# Using uv (faster)
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt
```

## Running Code
The modules are designed to be self-contained. You can run any algorithm script directly:
```bash
python 1-introduction/dynamic-prog/gridworld_case_study.py
python cases/1-gym_atari_lunar_lander/lunar_lander4_ddqn.py
```
Jupyter notebooks are also available for interactive exploration:
```bash
jupyter notebook 1-introduction/env.ipynb
```

## Development Conventions
- **Diagrams**: Mermaid is used for backup diagrams and flowcharts in markdown files. To ensure visibility in the portal, use standard GFM code blocks (```mermaid). The layout is configured to automatically convert these into renderable diagrams.

- **Environment API**: Custom environments follow the [Gymnasium API](https://gymnasium.farama.org/).
- **Neural Networks**: DQN implementations typically use a 3-layer fully-connected architecture (64 -> 64 hidden units) with `ReLU` activations.
- **Visualization**: Implementations often generate HTML dashboards with embedded plots for performance analysis.
- **State Management**: Tabular methods use a dictionary or numpy array for value functions/policies. Deep methods use PyTorch `nn.Module`.
- **Reproducibility**: No central test suite exists. Validation is performed by running scripts and observing convergence in rewards or visualization dashboards.

### Interactive Web Visualization: RL Methods Comparison
Located at `barto-sutton-notes/lecture3-mdp-dp/assets/policy-evaluation-dp/` — a React + Vite + TypeScript app demonstrating DP and model-free methods on a 4×4 gridworld with danger zones.

**Sections:**
- **MDP Environment**: Grid with states, action arrows, goal (🏁, R=+10), danger cells (💀, R=-5)
- **Policy Iteration (Method 1)**: Step 1 Policy Evaluation (3 policies: uniform, optimistic, suboptimal) → Step 2 Policy Improvement (argmax over Q-values with Q-value overlay)
- **Value Iteration (Method 2)**: Combined update V(s) = max_a Q(s,a), live Q-value table showing how max Q overwrites V and implicit greedy policy at each step
- **Monte Carlo (Method 3)**: First-visit MC with ε-greedy exploration, episode path visualization, Q-table with average returns
- **TD Methods (Method 4)**: SARSA (on-policy) and Q-Learning (off-policy) side-by-side, live Q-tables with best action indicators
- **Comparisons**: DP methods comparison + all-methods final policy comparison table

**Convergence criteria:**
- DP methods (Policy Eval, Value Iter): max|ΔV| < 0.001
- Model-free methods (MC, SARSA, Q-Learning): max|ΔQ| < 0.01 after 50+ episodes

**Running:**
```bash
cd barto-sutton-notes/lecture3-mdp-dp/assets/policy-evaluation-dp
npm install
npm run dev
```

**Config:** `GRID_SIZE=4`, `gamma=0.9`, `rewardGoal=10`, `rewardStep=0`, `DANGER_STATES={5,9}`, `REWARD_DANGER=-5`, `ALPHA=0.1`, `EPSILON=0.2`

## Key Implementation Patterns
- **Exploration**: Most agents use epsilon-greedy exploration with linear or exponential decay.
- **Buffers**: Deep RL implementations use a `ReplayBuffer` (found in `cases/1-gym_atari_lunar_lander/replay_buffer.py`).
- **Configs**: `GridWorldConfig` and similar dataclasses are used to configure environment and agent parameters.
