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
- **`class-discs/sara-sojourn/`**: Comparative dashboards and visualizations of different algorithms.
- **`books/memory/`**: Comprehensive notes on foundational texts like Sutton & Barto.
- **`assignments/`**: Course-related tasks and quizzes.

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
- **Environment API**: Custom environments follow the [Gymnasium API](https://gymnasium.farama.org/).
- **Neural Networks**: DQN implementations typically use a 3-layer fully-connected architecture (64 -> 64 hidden units) with `ReLU` activations.
- **Visualization**: Implementations often generate HTML dashboards with embedded plots for performance analysis.
- **State Management**: Tabular methods use a dictionary or numpy array for value functions/policies. Deep methods use PyTorch `nn.Module`.
- **Reproducibility**: No central test suite exists. Validation is performed by running scripts and observing convergence in rewards or visualization dashboards.

## Key Implementation Patterns
- **Exploration**: Most agents use epsilon-greedy exploration with linear or exponential decay.
- **Buffers**: Deep RL implementations use a `ReplayBuffer` (found in `cases/1-gym_atari_lunar_lander/replay_buffer.py`).
- **Configs**: `GridWorldConfig` and similar dataclasses are used to configure environment and agent parameters.
