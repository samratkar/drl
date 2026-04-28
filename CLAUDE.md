# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Deep Reinforcement Learning (DRL) course repository implementing algorithms from foundational methods (Dynamic Programming, Monte Carlo, TD Learning) through advanced deep learning approaches (DQN, DDQN, PPO, Actor-Critic). Built with Python 3.12, PyTorch, and Gymnasium.

## Setup & Running

```bash
# Python 3.12 required (.python-version specifies this)
source .venv/bin/activate

# Install dependencies (uv or pip)
uv pip install -r requirements.txt

# Run any module directly — each algorithm file is self-contained
python 1-introduction/dynamic-prog/gridworld_case_study.py
python cases/1-gym_atari_lunar_lander/lunar_lander4_ddqn.py

# Jupyter notebooks
jupyter notebook 1-introduction/env.ipynb
```

No test suite, linter, or build system — modules are run individually.

## Architecture

### Repository Organization

Numbered directories (1-17) form a sequential curriculum. Key implementation areas:

- `1-introduction/` — Core tabular RL: Dynamic Programming (`dynamic-prog/`), Monte Carlo (`montecarlo/`), custom environments (`assets/`)
- `cases/1-gym_atari_lunar_lander/` — Progressive DQN implementation chain: basic → replay buffer → fixed Q-targets → DDQN → prioritized replay (files `lunar_lander1` through `lunar_lander5`)
- `class-discs/sara-sojourn/assets/` — Algorithm comparison dashboards (DP, TD, MC, DQN) that generate self-contained HTML reports
- `assignments/` — Course assignments and quizzes

### Environment Pattern

Custom environments follow the Gymnasium API with a tabular transition model:

```python
env.reset() -> (state, info)
env.step(action) -> (next_state, reward, terminated, truncated, info)
env.P[state][action] -> [(probability, next_state, reward, done), ...]
```

`GridWorldConfig` dataclass drives environment configuration. Stochastic transitions are modeled via probability aggregation in the `P` table.

### Algorithm Pattern

Each algorithm implementation contains: core function (e.g., `policy_iteration()`, `train_dqn()`), epsilon-greedy exploration with decay, history/checkpoint tracking, and visualization output (HTML dashboards with embedded base64 matplotlib charts or JSON serialization).

### Neural Network Convention

DQN variants use a consistent 3-layer fully-connected architecture (state_dim → 64 → 64 → action_dim) with `QNetwork(nn.Module)`. Shared components (`q_network.py`, `replay_buffer.py`) live alongside the lunar lander implementations.

### Dependencies

PyTorch for neural networks, Gymnasium (with Box2D) for environments, numpy/matplotlib/seaborn for computation and visualization.
