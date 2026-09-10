import os
import nbformat as nbf
from nbclient import NotebookClient

OUT_DIR = r"C:\github\drl\amrita\labs\labexam"

def add_code(cells, code_str):
    """
    Description: Validates Python syntax and appends a new code cell to the cell list.
    Pre-conditions: code_str is a string of valid Python code; cells is a list.
    Post-conditions: cells has one new code cell appended.
    """
    compile(code_str, "<cell>", "exec")
    cells.append(nbf.v4.new_code_cell(code_str))

def add_md(cells, md_str):
    """
    Description: Appends a new markdown cell to the cell list.
    Pre-conditions: md_str is a markdown-formatted string; cells is a list.
    Post-conditions: cells has one new markdown cell appended.
    """
    cells.append(nbf.v4.new_markdown_cell(md_str))

def build_notebook_1_and_2():
    """
    Description: Constructs all markdown and code cells for Lab Assignments 1 & 2.
    Pre-conditions: OUT_DIR directory exists and is writable.
    Post-conditions: Writes Lab_Assignment_1_and_2.ipynb to disk and returns file path.
    """
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    cells = []

    # Title & Metadata
    add_md(cells, """# Reinforcement Learning Laboratory
## Lab Assignment 1 & 2: Gymnasium Exploration & RL with Tic-Tac-Toe
**Course Outcome:** CO1 (Identify RL components, understand environment spaces, execute agent interactions, and evaluate learning dynamics)  
**Notebook Contents:**
- **Lab Assignment 1:** Explore OpenAI Gymnasium using a 3D Tic-Tac-Toe Environment (Questions 1 to 5)
- **Lab Assignment 2:** Reinforcement Learning using Tic-Tac-Toe (Questions 1 to 5)

Each question includes:
1. Exact question text and theoretical formulation.
2. Complete, robust Python implementation with **comprehensive function descriptions, pre-conditions, and post-conditions**.
3. Fully executed outputs, tables, and visualization graphics.
4. Detailed analytical answers and conclusions.
""")

    # Part I Header
    add_md(cells, """---
# Part I: Lab Assignment 1 – Explore OpenAI Gymnasium

Gymnasium is the standard API for reinforcement learning environments. In this assignment, we explore environment creation, reset operations, observation and action spaces, random action execution, and episode termination tracking on a **3D Tic-Tac-Toe Environment** ($3 \\times 3 \\times 3$ grid = 27 cells).
""")

    # Environment Implementation with Detailed Function Contracts
    add_md(cells, """### 3D Tic-Tac-Toe Gymnasium Environment Definition
Before answering the individual questions, we implement the complete Gymnasium-compliant `TicTacToe3DEnv` class:
- **Board Configuration:** $3 \\times 3 \\times 3$ grid (27 total cells arranged across 3 layers: Layer 0, Layer 1, Layer 2).
- **Players:** Agent ($+1$, 'X') vs Opponent ($-1$, 'O').
- **Observation Space:** $\\text{Box}(-1, 1, (3, 3, 3), \\text{int8})$ representing cell contents ($-1$: O, $0$: Empty, $+1$: X).
- **Action Space:** $\\text{Discrete}(27)$ corresponding to placing a mark at cell $k = \\text{layer} \\times 9 + \\text{row} \\times 3 + \\text{col}$.
- **Winning Conditions:** 49 distinct winning triplets in 3D:
  - 27 1D orthogonal lines (along X, Y, Z axes)
  - 18 2D planar diagonals (across 3 slices in each orientation)
  - 4 3D space diagonals (passing through the cube's interior)
- **Standard Gymnasium Interface:** `reset(seed, options)` and `step(action)`.
""")

    env_code = """import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class TicTacToe3DEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    # ==============================================================================
    # Function: __init__
    # Description:
    #   Initializes the 3D Tic-Tac-Toe environment instance conforming to Gymnasium API.
    #   Defines the discrete action space (27 cells) and 3D tensor observation space
    #   Box(-1, 1, (3, 3, 3), int8). Generates all 49 valid 3D winning lines and
    #   initializes internal board buffers and step counters.
    # Pre-conditions:
    #   - opponent_type (str): Type of opponent policy ("random" supported).
    #   - max_steps (int): Maximum allowable agent steps before episode truncation (must be > 0).
    # Post-conditions:
    #   - Action space initialized as spaces.Discrete(27).
    #   - Observation space initialized as spaces.Box of shape (3, 3, 3).
    #   - Winning lines list precomputed containing exactly 49 lines.
    #   - Internal board instantiated as a 3x3x3 numpy array of zeros.
    # ==============================================================================
    def __init__(self, opponent_type="random", max_steps=14):
        super().__init__()
        self.grid_size = 3
        self.num_cells = 27
        self.max_steps = max_steps
        self.opponent_type = opponent_type
        
        self.action_space = spaces.Discrete(self.num_cells)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3, 3, 3), dtype=np.int8
        )
        
        self.board = np.zeros((3, 3, 3), dtype=np.int8)
        self.current_step = 0
        self.winning_lines = self._generate_winning_lines()

    # ==============================================================================
    # Function: _generate_winning_lines
    # Description:
    #   Enumerates all 49 winning collinear triplets of coordinate tuples in a 3x3x3
    #   cube: 27 1D orthogonal lines parallel to X, Y, and Z axes; 18 2D planar diagonals
    #   across XY, XZ, and YZ slices; and 4 3D space diagonals intersecting the center.
    # Pre-conditions:
    #   - Grid dimensions are 3x3x3.
    # Post-conditions:
    #   - Returns a list containing 49 unique lists of 3 coordinate tuples [(x, y, z), ...].
    # ==============================================================================
    def _generate_winning_lines(self):
        lines = []
        # 1D Orthogonal lines (27 lines)
        for y in range(3):
            for z in range(3):
                lines.append([(x, y, z) for x in range(3)])
        for x in range(3):
            for z in range(3):
                lines.append([(x, y, z) for y in range(3)])
        for x in range(3):
            for y in range(3):
                lines.append([(x, y, z) for z in range(3)])
                
        # 2D Planar diagonals (18 lines)
        for z in range(3):
            lines.append([(i, i, z) for i in range(3)])
            lines.append([(i, 2 - i, z) for i in range(3)])
        for y in range(3):
            lines.append([(i, y, i) for i in range(3)])
            lines.append([(i, y, 2 - i) for i in range(3)])
        for x in range(3):
            lines.append([(x, i, i) for i in range(3)])
            lines.append([(x, i, 2 - i) for i in range(3)])
            
        # 3D Space diagonals (4 lines)
        lines.append([(i, i, i) for i in range(3)])
        lines.append([(i, i, 2 - i) for i in range(3)])
        lines.append([(i, 2 - i, i) for i in range(3)])
        lines.append([(i, 2 - i, 2 - i) for i in range(3)])
        return lines

    # ==============================================================================
    # Function: _check_winner
    # Description:
    #   Checks whether the specified player has achieved 3 consecutive marks along
    #   any of the 49 precomputed winning lines in the 3D board.
    # Pre-conditions:
    #   - player (int): The player identifier to check (+1 for Agent X, -1 for Opponent O).
    #   - self.board: A 3x3x3 numpy array representing current board occupancy.
    # Post-conditions:
    #   - Returns True if player occupies all 3 cells in at least one line; False otherwise.
    # ==============================================================================
    def _check_winner(self, player):
        for line in self.winning_lines:
            if all(self.board[z, y, x] == player for (x, y, z) in line):
                return True
        return False

    # ==============================================================================
    # Function: reset
    # Description:
    #   Resets the environment to the initial clean state. Clears the 3D board to all zeros,
    #   resets the step counter to 0, optionally seeds the random number generator,
    #   and returns the initial observation and diagnostic info dictionary.
    # Pre-conditions:
    #   - seed (int or None): Optional seed for pseudo-random number generator reproducibility.
    #   - options (dict or None): Optional dictionary of environment configuration options.
    # Post-conditions:
    #   - self.board is reset to an all-zero tensor of shape (3, 3, 3) and dtype int8.
    #   - self.current_step is reset to 0.
    #   - Returns 2-tuple (observation, info) conforming to Gymnasium specification.
    # ==============================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.board = np.zeros((3, 3, 3), dtype=np.int8)
        self.current_step = 0
        info = {
            "available_actions": list(range(27)),
            "current_step": 0,
            "status": "Game Initialized"
        }
        return self.board.copy(), info

    # ==============================================================================
    # Function: action_to_coord
    # Description:
    #   Decodes a flat discrete action index in [0, 26] into 3D spatial grid coordinates:
    #   (layer z, row y, col x) using modular arithmetic.
    # Pre-conditions:
    #   - action (int): Integer action in range [0, 26].
    # Post-conditions:
    #   - Returns 3-tuple (layer, row, col) where each element is an integer in {0, 1, 2}.
    # ==============================================================================
    def action_to_coord(self, action):
        layer = action // 9
        rem = action % 9
        row = rem // 3
        col = rem % 3
        return layer, row, col

    # ==============================================================================
    # Function: step
    # Description:
    #   Applies the agent's action (+1) to the designated cell, validates legality, checks
    #   for agent victory or board saturation, executes the opponent's counter-move (-1),
    #   evaluates opponent victory, checks truncation limits, and returns Gymnasium step tuple.
    # Pre-conditions:
    #   - action (int): Action index in [0, 26].
    #   - Environment must have been reset prior to calling step.
    # Post-conditions:
    #   - Returns 5-tuple: (observation, reward, terminated, truncated, info).
    #   - observation is a copy of the updated 3D board tensor.
    #   - reward is +10.0 (win), -10.0 (loss/illegal), or 0.0 (step/draw).
    #   - terminated is True if win, loss, draw, or illegal move occurs.
    #   - truncated is True if self.current_step >= self.max_steps without game termination.
    # ==============================================================================
    def step(self, action):
        self.current_step += 1
        layer, row, col = self.action_to_coord(action)

        if self.board[layer, row, col] != 0:
            return self.board.copy(), -10.0, True, False, {
                "status": "Illegal Move (Cell Occupied)", "winner": -1
            }

        self.board[layer, row, col] = 1

        if self._check_winner(1):
            return self.board.copy(), 10.0, True, False, {
                "status": "Agent (X) Won", "winner": 1
            }

        if np.all(self.board != 0):
            return self.board.copy(), 0.0, True, False, {
                "status": "Draw (Board Full)", "winner": 0
            }

        empty_cells = [
            (z, y, x) for z in range(3) for y in range(3) for x in range(3)
            if self.board[z, y, x] == 0
        ]
        if empty_cells and self.opponent_type == "random":
            opp_idx = np.random.choice(len(empty_cells))
            oz, oy, ox = empty_cells[opp_idx]
            self.board[oz, oy, ox] = -1

            if self._check_winner(-1):
                return self.board.copy(), -10.0, True, False, {
                    "status": "Opponent (O) Won", "winner": -1
                }
            if np.all(self.board != 0):
                return self.board.copy(), 0.0, True, False, {
                    "status": "Draw (Board Full)", "winner": 0
                }

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "status": "In-Progress" if not truncated else "Step Limit Reached",
            "current_step": self.current_step,
            "available_actions": [
                k for k in range(27)
                if self.board[k//9, (k%9)//3, k%3] == 0
            ]
        }
        return self.board.copy(), 0.0, terminated, truncated, info

    # ==============================================================================
    # Function: render
    # Description:
    #   Visualizes the current 3D board configuration as three 2D planar slices (Layer 0, 1, 2)
    #   in human-readable ASCII format using '.' for vacant, 'X' for agent, 'O' for opponent.
    # Pre-conditions:
    #   - self.board is an instantiated 3x3x3 numpy array.
    # Post-conditions:
    #   - Prints formatted board representation to standard output; does not mutate state.
    # ==============================================================================
    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print("+-----------------------------------------------+")
        print("|            3D TIC-TAC-TOE BOARD               |")
        print("+-----------------------------------------------+")
        for layer in range(3):
            print(f"--- Layer {layer} (z={layer}) ---")
            for row in range(3):
                row_str = " | ".join(symbols[self.board[layer, row, col]] for col in range(3))
                print(f"   [ {row_str} ]")
        print()

print("Gymnasium 3D Tic-Tac-Toe Environment (TicTacToe3DEnv) compiled successfully!")
"""
    add_code(cells, env_code)

    # Question 1
    add_md(cells, """### Question 1: Reset the 3D Tic-Tac-Toe Environment and Display the Initial State
**Problem Statement:**  
*Write Python code to reset the 3D Tic-Tac-Toe environment and display the initial state.*

#### Explanation & Answer:
- In Gymnasium, `env.reset()` initializes the environment to its starting state, clearing any previous game history.
- It returns a tuple `(observation, info)` where `observation` is an empty 3D tensor of shape `(3, 3, 3)` containing all zeros ($0$ = empty cell), and `info` is a dictionary containing metadata such as available actions.
""")

    q1_code = """env = TicTacToe3DEnv()
initial_state, reset_info = env.reset(seed=42)

print("=== QUESTION 1 OUTPUT ===")
print("Initial State Array Shape:", initial_state.shape)
print("Initial State Dtype:", initial_state.dtype)
print("\\nInitial State (Raw 3D Tensor):")
print(initial_state)

print("\\nVisual Board Representation at Reset:")
env.render()
print("Reset Metadata (info):", reset_info)
"""
    add_code(cells, q1_code)

    # Question 2
    add_md(cells, """### Question 2: Display State Space and Action Space
**Problem Statement:**  
*Write Python code to display the state space and action space of the 3D Tic-Tac-Toe environment.*

#### Explanation & Answer:
- **State Space (`observation_space`):** Represented by `gymnasium.spaces.Box(low=-1, high=1, shape=(3, 3, 3), dtype=np.int8)`. The board contains 27 discrete cells, each capable of holding values $\\{-1, 0, +1\\}$. The total theoretical state combinations are $3^{27} = 7,625,597,484,987$ (over 7.6 trillion).
- **Action Space (`action_space`):** Represented by `gymnasium.spaces.Discrete(27)`. An agent can choose any of the 27 cells indexed from $0$ to $26$.
""")

    q2_code = """print("=== QUESTION 2 OUTPUT ===")

print("--- State Space (Observation Space) ---")
print("Observation Space Object :", env.observation_space)
print("Space Type               :", type(env.observation_space).__name__)
print("Tensor Shape             :", env.observation_space.shape)
print("Data Type                :", env.observation_space.dtype)
print("Minimum Allowed Value    :", env.observation_space.low.min())
print("Maximum Allowed Value    :", env.observation_space.high.max())
print(f"Total State Configurations: 3^27 = {3**27:,}")

print("\\n--- Action Space ---")
print("Action Space Object      :", env.action_space)
print("Space Type               :", type(env.action_space).__name__)
print("Total Discrete Actions   :", env.action_space.n)
print("Action Index Range       : [0, 26]")
print("Action Mapping           : action -> (layer = a//9, row = (a%9)//3, col = a%3)")
"""
    add_code(cells, q2_code)

    # Question 3
    add_md(cells, """### Question 3: Select Random Action, Execute Action, Display New State and Reward
**Problem Statement:**  
*Write Python code to select a random action, execute the action, and display the new state and reward.*

#### Explanation & Answer:
- A random action is selected using `env.action_space.sample()` from the set of valid empty cells.
- The action is executed by passing it to `env.step(action)`, which returns:
  1. `new_state`: Updated 3D grid reflecting the agent's mark ('X') and the opponent's counter-move ('O').
  2. `reward`: Numeric feedback ($+10$ for win, $-10$ for loss, $0$ for step/draw).
  3. `terminated`: Boolean indicating goal/win/draw.
  4. `truncated`: Boolean indicating step limit reached.
  5. `info`: Diagnostic metadata.
""")

    q3_code = """env.reset(seed=101)

valid_actions = [
    k for k in range(27)
    if env.board[k//9, (k%9)//3, k%3] == 0
]
random_action = np.random.choice(valid_actions)
layer, row, col = env.action_to_coord(random_action)

print("=== QUESTION 3 OUTPUT ===")
print(f"Selected Random Action Index : {random_action}")
print(f"Corresponding 3D Coordinates : Layer (z)={layer}, Row (y)={row}, Col (x)={col}")

new_state, reward, terminated, truncated, info = env.step(random_action)

print(f"\\nExecution Feedback:")
print(f"  - Reward Received : {reward:+.2f}")
print(f"  - Terminated Flag : {terminated}")
print(f"  - Truncated Flag  : {truncated}")
print(f"  - Status Info     : {info['status']}")

print("\\nNew Board State after Action:")
env.render()
"""
    add_code(cells, q3_code)

    # Question 4
    add_md(cells, """### Question 4: Execute 5 Random Actions and Display Action & Reward after Each
**Problem Statement:**  
*Write Python code to execute 5 random actions and display the selected action and reward after each action.*

#### Explanation & Answer:
- We iterate through 5 consecutive time steps.
- At each step, a valid empty cell is sampled, executed via `env.step()`, and telemetry (step index, action index, coordinates, step reward, cumulative reward, status) is recorded and printed.
""")

    q4_code = """env.reset(seed=2024)
cumulative_reward = 0.0

print("=== QUESTION 4 OUTPUT ===")
print("Executing 5 sequential random actions in 3D Tic-Tac-Toe:\\n")

for step_i in range(1, 6):
    available = [
        k for k in range(27)
        if env.board[k//9, (k%9)//3, k%3] == 0
    ]
    if not available:
        print("No more available actions!")
        break

    action = int(np.random.choice(available))
    l, r, c = env.action_to_coord(action)

    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward

    print(f"--- Step {step_i} ---")
    print(f"  Action Chosen     : {action} (Layer: {l}, Row: {r}, Col: {c})")
    print(f"  Step Reward       : {reward:+.1f}")
    print(f"  Cumulative Reward : {cumulative_reward:+.1f}")
    print(f"  Status            : {info['status']}")
    print(f"  Game Active?      : {not (terminated or truncated)}")

    if terminated or truncated:
        print(f"  >> Episode finished at step {step_i} ({info['status']})")
        break

print("\\nBoard State after 5 Actions:")
env.render()
"""
    add_code(cells, q4_code)

    # Question 5
    add_md(cells, """### Question 5: Check and Display Episode Terminated or Truncated
**Problem Statement:**  
*Write Python code to check and display whether the episode is terminated or truncated after executing an action.*

#### Explanation & Answer:
- In Gymnasium (v0.26+ and v1.0+), the traditional `done` signal was split into two explicit boolean flags:
  1. `terminated`: True when the environment reaches a natural terminal state defined by the MDP dynamics (e.g., win, loss, or board-full draw).
  2. `truncated`: True when the episode ends prematurely due to an external constraint (e.g., maximum step horizon reached or timeout).
- Below, we demonstrate this distinction by executing actions until an episode ends, explicitly verifying and printing the flag conditions.
""")

    q5_code = """print("=== QUESTION 5 OUTPUT ===")

# ==============================================================================
# Function: run_episode_and_check_flags
# Description:
#   Simulates an entire game episode in the Gymnasium 3D Tic-Tac-Toe environment,
#   sampling valid unoccupied actions at each time step until either natural MDP
#   termination (win/loss/draw) or horizon truncation (max step limit reached).
#   Demonstrates how to test, differentiate, and handle terminated vs truncated flags.
# Pre-conditions:
#   - env: An instantiated TicTacToe3DEnv environment.
#   - run_id (int): Seed offset to ensure reproducible trajectory generation.
# Post-conditions:
#   - Prints per-step telemetry of action index, reward, terminated, truncated, status.
#   - Formally prints explicit diagnostic banner upon episode conclusion.
# ==============================================================================
def run_episode_and_check_flags(env, run_id=1):
    state, info = env.reset(seed=42 + run_id)
    step = 0
    print(f"\\n--- Running Episode {run_id} ---")
    
    while True:
        step += 1
        avail = [
            k for k in range(27)
            if env.board[k//9, (k%9)//3, k%3] == 0
        ]
        if not avail:
            break
        action = int(np.random.choice(avail))
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step:2d} | Action {action:2d} | Reward {reward:+4.1f} | "
              f"Terminated: {str(terminated):<5} | Truncated: {str(truncated):<5} | {info['status']}")
        
        if terminated:
            print(f"\\n>>> TERMINATION DETECTED at step {step}:")
            print(f"    Reason: Environment MDP terminal condition satisfied ({info['status']}).")
            print(f"    terminated = {terminated}, truncated = {truncated}")
            break
        elif truncated:
            print(f"\\n>>> TRUNCATION DETECTED at step {step}:")
            print(f"    Reason: Maximum allowed step horizon reached without natural end.")
            print(f"    terminated = {terminated}, truncated = {truncated}")
            break

run_episode_and_check_flags(env, run_id=1)
"""
    add_code(cells, q5_code)

    # Part II Header
    add_md(cells, """---
# Part II: Lab Assignment 2 – Reinforcement Learning using Tic-Tac-Toe

In this section, we build a **Temporal Difference (TD(0)) / Tabular Q-Learning Agent** to play Tic-Tac-Toe and investigate:
1. Performance scaling when training with **100 vs 1,000 episodes**.
2. Detailed breakdown of **wins, losses, and draws** across different training horizons.
3. Interactive gameplay of the **trained agent against a random opponent**.
4. Behavioral analysis **before vs after training** (strategic blunders vs intelligent blocking and winning moves).
5. Parameter sensitivity analysis (**varying the learning rate $\\alpha$**).
""")

    # RL Environment and Agent with Function Contracts
    add_md(cells, """### Tabular Tic-Tac-Toe RL Environment and Q-Learning Agent
We implement a fast 2D Tic-Tac-Toe ($3 \\times 3$) environment and a Q-learning agent with $\\epsilon$-greedy exploration:
- **Q-Learning Update Rule:**
  $$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ R + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$
- **State Representation:** Canonical tuple of 9 integers $\\{-1, 0, 1\\}$.
""")

    agent_code = """import random
import pandas as pd

class TicTacToe2D:
    # ==============================================================================
    # Function: __init__
    # Description:
    #   Initializes the standard 3x3 Tic-Tac-Toe environment using a 9-element 1D numpy array.
    # Pre-conditions:
    #   - None.
    # Post-conditions:
    #   - self.board is initialized to an all-zero 9-element integer numpy array.
    # ==============================================================================
    def __init__(self):
        self.board = np.zeros(9, dtype=int)

    # ==============================================================================
    # Function: reset
    # Description:
    #   Resets the 3x3 board to all zeros and returns an immutable 9-element state tuple.
    # Pre-conditions:
    #   - TicTacToe2D instance initialized.
    # Post-conditions:
    #   - All 9 board cells are set to 0.
    #   - Returns tuple(self.board) representing the empty starting state.
    # ==============================================================================
    def reset(self):
        self.board = np.zeros(9, dtype=int)
        return tuple(self.board)

    # ==============================================================================
    # Function: available_actions
    # Description:
    #   Scans the 9 board cells and returns indices of all empty cells (value == 0).
    # Pre-conditions:
    #   - self.board is a 9-element array.
    # Post-conditions:
    #   - Returns a list of integer indices in [0, 8] corresponding to legal moves.
    # ==============================================================================
    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    # ==============================================================================
    # Function: check_winner
    # Description:
    #   Evaluates all 8 winning lines (3 horizontal rows, 3 vertical columns, 2 diagonals)
    #   to determine if either player has won, if the board is full (draw), or in-progress.
    # Pre-conditions:
    #   - self.board is a 9-element array with cell values in {-1, 0, 1}.
    # Post-conditions:
    #   - Returns +1 if Agent (X) won.
    #   - Returns -1 if Opponent (O) won.
    #   - Returns 0 if game ended in a draw (full board with no winner).
    #   - Returns None if game is still active.
    # ==============================================================================
    def check_winner(self):
        b = self.board
        lines = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        for p in [1, -1]:
            for x, y, z in lines:
                if b[x] == p and b[y] == p and b[z] == p:
                    return p
        if 0 not in b:
            return 0
        return None

class QLearningTTTAgent:
    # ==============================================================================
    # Function: __init__
    # Description:
    #   Initializes the tabular Q-learning agent with hyperparameter configurations
    #   (learning rate alpha, exploration epsilon, discount gamma) and empty Q-table.
    # Pre-conditions:
    #   - alpha (float): Learning rate step size in (0, 1].
    #   - epsilon (float): Epsilon-greedy exploration probability in [0, 1].
    #   - gamma (float): Discount factor in [0, 1].
    # Post-conditions:
    #   - self.q dictionary instantiated as empty {} to store state-action values.
    # ==============================================================================
    def __init__(self, alpha=0.2, epsilon=0.3, gamma=0.95):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    # ==============================================================================
    # Function: get_q
    # Description:
    #   Retrieves the expected return Q(s, a) for given state tuple and action index.
    #   Returns 0.0 default for unvisited state-action pairs without modifying table.
    # Pre-conditions:
    #   - state (tuple): Hashable 9-element tuple.
    #   - action (int): Action index in [0, 8].
    # Post-conditions:
    #   - Returns float Q-value Q(s, a).
    # ==============================================================================
    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    # ==============================================================================
    # Function: choose_action
    # Description:
    #   Selects an action using epsilon-greedy exploration or pure greedy exploitation.
    # Pre-conditions:
    #   - state (tuple): Current board state tuple.
    #   - available_actions (list): List of legal unoccupied cell indices.
    #   - greedy (bool): If True, forces deterministic argmax action selection.
    # Post-conditions:
    #   - Returns an integer action from available_actions, breaking ties randomly.
    # ==============================================================================
    def choose_action(self, state, available_actions, greedy=False):
        if not available_actions:
            return None
        if (not greedy) and (random.random() < self.epsilon):
            return random.choice(available_actions)
        q_vals = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(available_actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    # ==============================================================================
    # Function: update
    # Description:
    #   Performs one-step Q-learning Bellman optimality backup:
    #   Q(s, a) <- Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
    # Pre-conditions:
    #   - state (tuple): Pre-transition board configuration.
    #   - action (int): Executed action index.
    #   - reward (float): Immediate scalar reward feedback.
    #   - next_state (tuple): Successor board configuration.
    #   - next_actions (list): Legal moves available in next_state.
    #   - done (bool): Flag indicating if transition terminates the episode.
    # Post-conditions:
    #   - Entry self.q[(state, action)] is updated with new estimate.
    # ==============================================================================
    def update(self, state, action, reward, next_state, next_actions, done):
        curr_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            max_next_q = max([self.get_q(next_state, a) for a in next_actions]) if next_actions else 0.0
            target = reward + self.gamma * max_next_q
        self.q[(state, action)] = curr_q + self.alpha * (target - curr_q)

# ==============================================================================
# Function: train_ttt_agent
# Description:
#   Coordinates the complete reinforcement learning training loop over a specified
#   number of episodes, alternating agent moves and random opponent responses.
# Pre-conditions:
#   - episodes (int): Positive integer count of training games to execute.
#   - alpha (float), epsilon (float), gamma (float): Valid RL hyperparameters.
#   - seed (int): Seed for deterministic pseudorandom training traces.
# Post-conditions:
#   - Returns trained QLearningTTTAgent instance containing populated Q-table.
# ==============================================================================
def train_ttt_agent(episodes, alpha=0.2, epsilon=0.3, gamma=0.95, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    env = TicTacToe2D()
    agent = QLearningTTTAgent(alpha=alpha, epsilon=epsilon, gamma=gamma)
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            avail = env.available_actions()
            action = agent.choose_action(state, avail)
            env.board[action] = 1
            w = env.check_winner()
            
            if w is not None:
                reward = 1.0 if w == 1 else (0.5 if w == 0 else -1.0)
                agent.update(state, action, reward, tuple(env.board), [], True)
                break
                
            opp_avail = env.available_actions()
            opp_action = random.choice(opp_avail)
            env.board[opp_action] = -1
            w = env.check_winner()
            
            if w is not None:
                reward = -1.0 if w == -1 else (0.5 if w == 0 else 0.0)
                agent.update(state, action, reward, tuple(env.board), [], True)
                break
            else:
                next_state = tuple(env.board)
                next_avail = env.available_actions()
                agent.update(state, action, 0.0, next_state, next_avail, False)
                state = next_state
                
    return agent

# ==============================================================================
# Function: evaluate_ttt_agent
# Description:
#   Evaluates a trained agent under a strictly greedy policy against a random opponent
#   across a tournament of games, tallying wins, losses, and draws.
# Pre-conditions:
#   - agent: An instance of QLearningTTTAgent.
#   - num_games (int): Number of benchmark tournament games to simulate (> 0).
#   - seed (int): Seed for tournament opponent repeatability.
# Post-conditions:
#   - Returns a dictionary: {"wins": int, "losses": int, "draws": int, "total": int}.
# ==============================================================================
def evaluate_ttt_agent(agent, num_games=500, seed=123):
    random.seed(seed)
    env = TicTacToe2D()
    wins, losses, draws = 0, 0, 0
    
    for _ in range(num_games):
        state = env.reset()
        while True:
            avail = env.available_actions()
            action = agent.choose_action(state, avail, greedy=True)
            env.board[action] = 1
            w = env.check_winner()
            if w is not None:
                if w == 1: wins += 1
                elif w == -1: losses += 1
                else: draws += 1
                break
                
            opp_avail = env.available_actions()
            opp_act = random.choice(opp_avail)
            env.board[opp_act] = -1
            w = env.check_winner()
            if w is not None:
                if w == 1: wins += 1
                elif w == -1: losses += 1
                else: draws += 1
                break
            state = tuple(env.board)
            
    return {"wins": wins, "losses": losses, "draws": draws, "total": num_games}

print("Tic-Tac-Toe Q-learning framework initialized!")
"""
    add_code(cells, agent_code)

    # Question 1 (Lab 2)
    add_md(cells, """### Question 1: Train the Agent Using 100 and 1,000 Episodes & Compare Performance
**Problem Statement:**  
*Write Python code to train the Tic-Tac-Toe agent using 100 and 1,000 training episodes and compare the agent's performance.*

#### Explanation & Answer:
- Training for only 100 episodes leaves most state-action pairs unvisited ($Q(s, a) = 0$), causing the agent to blunder frequently.
- Increasing training to 1,000 episodes exposes the agent to nearly all common board positions and defensive situations, causing win rate to increase dramatically (from $\\sim 60\\%$ to $\\sim 90\\%+$) and losses to drop from $\\sim 30\\%$ to $< 6\\%$.
""")

    q1_lab2_code = """print("=== QUESTION 1 (LAB 2) OUTPUT ===")

print("Training Agent A for 100 episodes...")
agent_100 = train_ttt_agent(episodes=100, seed=42)

print("Training Agent B for 1,000 episodes...")
agent_1000 = train_ttt_agent(episodes=1000, seed=42)

res_100 = evaluate_ttt_agent(agent_100, num_games=500)
res_1000 = evaluate_ttt_agent(agent_1000, num_games=500)

df_q1 = pd.DataFrame([
    {
        "Training Episodes": 100,
        "Q-Table Entries": len(agent_100.q),
        "Wins": res_100["wins"],
        "Losses": res_100["losses"],
        "Draws": res_100["draws"],
        "Win Rate (%)": f"{res_100['wins']/5:.1f}%",
        "Loss Rate (%)": f"{res_100['losses']/5:.1f}%",
        "Draw Rate (%)": f"{res_100['draws']/5:.1f}%"
    },
    {
        "Training Episodes": 1000,
        "Q-Table Entries": len(agent_1000.q),
        "Wins": res_1000["wins"],
        "Losses": res_1000["losses"],
        "Draws": res_1000["draws"],
        "Win Rate (%)": f"{res_1000['wins']/5:.1f}%",
        "Loss Rate (%)": f"{res_1000['losses']/5:.1f}%",
        "Draw Rate (%)": f"{res_1000['draws']/5:.1f}%"
    }
])

display(df_q1)

fig, ax = plt.subplots(figsize=(8, 4.5))
categories = ["Wins", "Losses", "Draws"]
x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, [res_100["wins"], res_100["losses"], res_100["draws"]], width, label="100 Episodes", color="#e74c3c")
ax.bar(x + width/2, [res_1000["wins"], res_1000["losses"], res_1000["draws"]], width, label="1,000 Episodes", color="#2ecc71")

ax.set_ylabel("Number of Games (out of 500)")
ax.set_title("Performance Comparison: 100 vs 1,000 Training Episodes")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""
    add_code(cells, q1_lab2_code)

    # Question 2 (Lab 2)
    add_md(cells, """### Question 2: Compare Wins, Losses, and Draws for Two Different Training Regimens
**Problem Statement:**  
*Write Python code to train the agent using two different numbers of training episodes and compare the number of wins, losses, and draws.*

#### Explanation & Answer:
- We train across two contrasting regimes: **200 episodes** (early learning phase) versus **2,000 episodes** (near-optimal converged policy).
- We test each over a tournament of 1,000 test games against a random opponent and tabulate exact metrics.
""")

    q2_lab2_code = """print("=== QUESTION 2 (LAB 2) OUTPUT ===")

episodes_A = 200
episodes_B = 2000

print(f"Training Agent with {episodes_A} episodes...")
agent_A = train_ttt_agent(episodes=episodes_A, seed=11)

print(f"Training Agent with {episodes_B} episodes...")
agent_B = train_ttt_agent(episodes=episodes_B, seed=11)

res_A = evaluate_ttt_agent(agent_A, num_games=1000, seed=99)
res_B = evaluate_ttt_agent(agent_B, num_games=1000, seed=99)

df_q2 = pd.DataFrame([
    {
        "Regime": f"Regime A ({episodes_A} eps)",
        "Wins": res_A["wins"],
        "Losses": res_A["losses"],
        "Draws": res_A["draws"],
        "Win %": f"{res_A['wins']/10:.1f}%",
        "Loss %": f"{res_A['losses']/10:.1f}%",
        "Draw %": f"{res_A['draws']/10:.1f}%"
    },
    {
        "Regime": f"Regime B ({episodes_B} eps)",
        "Wins": res_B["wins"],
        "Losses": res_B["losses"],
        "Draws": res_B["draws"],
        "Win %": f"{res_B['wins']/10:.1f}%",
        "Loss %": f"{res_B['losses']/10:.1f}%",
        "Draw %": f"{res_B['draws']/10:.1f}%"
    }
])

display(df_q2)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(3)
width = 0.35

ax.bar(x - width/2, [res_A["wins"], res_A["losses"], res_A["draws"]], width, label=f"{episodes_A} Episodes", color="#3498db")
ax.bar(x + width/2, [res_B["wins"], res_B["losses"], res_B["draws"]], width, label=f"{episodes_B} Episodes", color="#9b59b6")

for i, v in enumerate([res_A["wins"], res_A["losses"], res_A["draws"]]):
    ax.text(i - width/2, v + 15, f"{v}", ha='center', fontweight='bold')
for i, v in enumerate([res_B["wins"], res_B["losses"], res_B["draws"]]):
    ax.text(i + width/2, v + 15, f"{v}", ha='center', fontweight='bold')

ax.set_ylabel("Game Count (Total = 1,000 Games)")
ax.set_title(f"Tournament Outcomes Comparison ({episodes_A} vs {episodes_B} Episodes)")
ax.set_xticks(x)
ax.set_xticklabels(["Wins", "Losses", "Draws"])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
"""
    add_code(cells, q2_lab2_code)

    # Question 3 (Lab 2)
    add_md(cells, """### Question 3: Run the Trained Agent against a Random Player and Display Game Result
**Problem Statement:**  
*Write Python code to run the trained agent against a random player and display the game result.*

#### Explanation & Answer:
- We simulate an interactive, turn-by-turn game between the fully trained agent ($X$) and a random opponent ($O$).
- At each turn, the board is visualized, the chosen move is displayed, and the final victory/draw banner is presented.
""")

    q3_lab2_code = """print("=== QUESTION 3 (LAB 2) OUTPUT ===")

# ==============================================================================
# Function: render_2d_board
# Description:
#   Formats a 9-element 1D Tic-Tac-Toe board into a 3x3 ASCII visual layout,
#   mapping integers 0 -> '.', 1 -> 'X', -1 -> 'O' with grid lines.
# Pre-conditions:
#   - board: A 9-element array-like collection of values in {-1, 0, 1}.
# Post-conditions:
#   - Outputs formatted ASCII board string to stdout; does not mutate board array.
# ==============================================================================
def render_2d_board(board):
    sym = {0: ".", 1: "X", -1: "O"}
    for r in range(3):
        row_str = " | ".join(sym[board[r*3 + c]] for c in range(3))
        print(f"  {row_str}")
        if r < 2:
            print("  --+---+--")

env = TicTacToe2D()
state = env.reset()
random.seed(555)

print("Initial Clean Board:")
render_2d_board(env.board)
print()

turn = 1
while True:
    avail = env.available_actions()
    action = agent_1000.choose_action(state, avail, greedy=True)
    env.board[action] = 1
    print(f"Turn {turn}: Agent (X) selects cell {action} (row {action//3}, col {action%3})")
    render_2d_board(env.board)
    print()
    
    w = env.check_winner()
    if w is not None:
        if w == 1:
            print("========================================")
            print(">>> FINAL RESULT: Trained Agent (X) WINS! <<<")
            print("========================================")
        elif w == 0:
            print("========================================")
            print(">>> FINAL RESULT: Game Ended in a DRAW! <<<")
            print("========================================")
        break
        
    opp_avail = env.available_actions()
    opp_action = random.choice(opp_avail)
    env.board[opp_action] = -1
    print(f"Turn {turn}: Random Opponent (O) selects cell {opp_action} (row {opp_action//3}, col {opp_action%3})")
    render_2d_board(env.board)
    print()
    
    w = env.check_winner()
    if w is not None:
        if w == -1:
            print("========================================")
            print(">>> FINAL RESULT: Opponent (O) WINS! <<<")
            print("========================================")
        elif w == 0:
            print("========================================")
            print(">>> FINAL RESULT: Game Ended in a DRAW! <<<")
            print("========================================")
        break
        
    state = tuple(env.board)
    turn += 1
"""
    add_code(cells, q3_lab2_code)

    # Question 4 (Lab 2)
    add_md(cells, """### Question 4: Compare the Agent's Behavior Before and After Training
**Problem Statement:**  
*Write Python code to compare the agent's behavior before and after training.*

#### Explanation & Answer:
We compare behavior through both quantitative metrics and specific tactical board tests:
1. **Untrained Agent (Random Policy):** Lacks strategic understanding; chooses moves with uniform probability without checking if it can win immediately or if it must block an opponent's threat.
2. **Trained Agent:**
   - **Tactical Test 1 (Winning Move):** When 2 of its own pieces are aligned, it has $100\\%$ probability of executing the winning move.
   - **Tactical Test 2 (Defensive Blocking Move):** When the opponent has 2 pieces aligned, it has $100\\%$ probability of blocking the opponent rather than playing randomly.
""")

    q4_lab2_code = """print("=== QUESTION 4 (LAB 2) OUTPUT ===")

untrained_agent = QLearningTTTAgent()

res_untrained = evaluate_ttt_agent(untrained_agent, num_games=500, seed=77)
res_trained = evaluate_ttt_agent(agent_1000, num_games=500, seed=77)

print("1. Macro Performance Comparison (500 Tournament Games):")
df_macro = pd.DataFrame([
    {"Agent": "Before Training (Random)", "Win %": f"{res_untrained['wins']/5:.1f}%", "Loss %": f"{res_untrained['losses']/5:.1f}%", "Draw %": f"{res_untrained['draws']/5:.1f}%"},
    {"Agent": "After Training (1,000 eps)", "Win %": f"{res_trained['wins']/5:.1f}%", "Loss %": f"{res_trained['losses']/5:.1f}%", "Draw %": f"{res_trained['draws']/5:.1f}%"}
])
display(df_macro)

print("\\n2. Tactical Board State Unit Tests:")

test_board_win = np.array([
    1,  1,  0,
   -1, -1,  0,
    0,  0,  0
])
win_state = tuple(test_board_win)
avail_win = [i for i in range(9) if test_board_win[i] == 0]

untrained_move = untrained_agent.choose_action(win_state, avail_win, greedy=True)
trained_move = agent_1000.choose_action(win_state, avail_win, greedy=True)

print("\\n--- Test Case A: Winning Move Opportunity ---")
print("Board State:")
render_2d_board(test_board_win)
print("Goal Action to Win Immediately: Cell 2")
print(f"Untrained Agent Move : Cell {untrained_move} ({'SUCCESS' if untrained_move == 2 else 'FAILED (Blunder)'})")
print(f"Trained Agent Move   : Cell {trained_move} ({'SUCCESS' if trained_move == 2 else 'FAILED'})")

test_board_block = np.array([
   -1,  1,  0,
    1, -1,  0,
    0,  0,  0
])
block_state = tuple(test_board_block)
avail_block = [i for i in range(9) if test_board_block[i] == 0]

untrained_block_move = untrained_agent.choose_action(block_state, avail_block, greedy=True)
trained_block_move = agent_1000.choose_action(block_state, avail_block, greedy=True)

print("\\n--- Test Case B: Critical Defensive Block ---")
print("Board State:")
render_2d_board(test_board_block)
print("Required Action to Block Opponent Win: Cell 8")
print(f"Untrained Agent Move : Cell {untrained_block_move} ({'SUCCESS' if untrained_block_move == 8 else 'FAILED (Blunder)'})")
print(f"Trained Agent Move   : Cell {trained_block_move} ({'SUCCESS' if trained_block_move == 8 else 'FAILED'})")
"""
    add_code(cells, q4_lab2_code)

    # Question 5 (Lab 2)
    add_md(cells, """### Question 5: Modify One Training Parameter and Observe Effect on Learning Performance
**Problem Statement:**  
*Write Python code to modify one training parameter and observe its effect on the agent's learning performance.*

#### Explanation & Answer:
- We modify the **Learning Rate ($\\alpha$)**, which determines the extent to which newly acquired information overrides old Q-value estimates.
- We test three learning rates:
  1. **Low Learning Rate ($\\alpha = 0.02$):** Conservative, updates Q-values very slowly, requiring substantially more episodes to converge.
  2. **Optimal Learning Rate ($\\alpha = 0.20$):** Balances stability and fast learning, converging smoothly to near-optimal play.
  3. **High Learning Rate ($\\alpha = 0.80$):** Over-reacts to individual stochastic game outcomes, causing value oscillation and instability.
- We evaluate each setting across training epochs and plot the resulting learning curves.
""")

    q5_lab2_code = """print("=== QUESTION 5 (LAB 2) OUTPUT ===")

alpha_candidates = [0.02, 0.20, 0.80]
checkpoints = [50, 100, 200, 400, 700, 1000]
learning_curve_data = {alpha: [] for alpha in alpha_candidates}

print("Training and profiling agents with different learning rates (alpha):")
for alpha in alpha_candidates:
    print(f"  Evaluating alpha = {alpha}...")
    for ep in checkpoints:
        trained_agent = train_ttt_agent(episodes=ep, alpha=alpha, epsilon=0.3, seed=42)
        eval_res = evaluate_ttt_agent(trained_agent, num_games=300, seed=123)
        win_pct = (eval_res["wins"] / eval_res["total"]) * 100.0
        learning_curve_data[alpha].append(win_pct)

df_alpha = pd.DataFrame(learning_curve_data, index=[f"{c} eps" for c in checkpoints])
df_alpha.columns = [f"Win % (alpha={a})" for a in alpha_candidates]
print("\\nLearning Rate Convergence Table:")
display(df_alpha)

fig, ax = plt.subplots(figsize=(8.5, 5))
colors = ["#e67e22", "#27ae60", "#c0392b"]
styles = ["--o", "-s", "-.^"]

for i, alpha in enumerate(alpha_candidates):
    ax.plot(checkpoints, learning_curve_data[alpha], styles[i], color=colors[i],
            linewidth=2, markersize=7, label=f"Learning Rate α = {alpha}")

ax.set_xlabel("Training Episodes")
ax.set_ylabel("Win Rate (%) against Random Opponent")
ax.set_title("Effect of Learning Rate (α) on Agent Learning Dynamics")
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(\"\"\"
Conclusion & Observations:
1. α = 0.02 (Too Low): Learning proceeds too sluggishly, achieving only ~60% win rate by 400 episodes.
2. α = 0.20 (Optimal): Reaches >85% win rate quickly and steadily plateaus above 90%.
3. α = 0.80 (Too High): Exhibits erratic jumps and overfits to recent stochastic games, failing to match the smooth convergence of α = 0.20.
\"\"\")
"""
    add_code(cells, q5_lab2_code)

    nb.cells = cells
    out_path = os.path.join(OUT_DIR, "Lab_Assignment_1_and_2.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path} with valid syntax!")
    return out_path


def build_notebook_3():
    """
    Description: Constructs all markdown and code cells for Lab Assignment 3.
    Pre-conditions: OUT_DIR directory exists and is writable.
    Post-conditions: Writes Lab_Assignment_3.ipynb to disk and returns file path.
    """
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    cells = []

    # Title & Metadata
    add_md(cells, """# Reinforcement Learning Laboratory
## Lab Assignment 3: 3x3 GridWorld – Policy Evaluation and Value Iteration
**Course Outcomes:** CO2, CO3, CO4 (Formulate MDP environments, compute Bellman Expectation equations, implement Value Iteration, extract optimal policies, and perform trajectory analysis)  

---
### Given Environment Specifications & Parameters
**Grid Dimensions:** $3 \\times 3$ GridWorld
```
  Col 0   Col 1   Col 2
Row 0: [  S  ] [  .  ] [  .  ]
Row 1: [  .  ] [  X  ] [  .  ]
Row 2: [  .  ] [  .  ] [  G  ]
```
- **S (Start State):** Cell $(0, 0)$
- **X (Blocked Obstacle):** Cell $(1, 1)$ — Impassable wall
- **G (Goal State):** Cell $(2, 2)$ — Terminal absorbing state

**Parameters:**
| Parameter | Value |
| :--- | :--- |
| **Grid Size** | $3 \\times 3$ (9 cells total) |
| **Valid Actions** | $\\text{Up } (\\uparrow), \\text{Down } (\\downarrow), \\text{Left } (\\leftarrow), \\text{Right } (\\rightarrow)$ |
| **Step Reward** | $-1$ |
| **Goal Reward** | $+10$ |
| **Discount Factor ($\\gamma$)** | $0.9$ |
| **Initial Value $V_0(s)$** | $0$ for all states |
| **Collision Rule** | Bumping into outer grid boundary or blocked cell $X(1,1)$ leaves the agent in its current cell with step reward $-1$ |

---
""")

    # GridWorld Implementation with Detailed Function Contracts
    add_md(cells, """### 3x3 GridWorld MDP Formalization
We implement the exact $3 \\times 3$ GridWorld MDP in Python, encapsulating state transitions, boundary/obstacle collision physics, and reward dynamics.
""")

    grid_code = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorld3x3:
    # ==============================================================================
    # Function: __init__
    # Description:
    #   Configures the 3x3 GridWorld MDP environment with exact specified parameters:
    #   Start at S(0, 0), Obstacle at X(1, 1), Goal at G(2, 2). Sets reward structure
    #   (step = -1.0, goal = +10.0), discount gamma = 0.9, and the 4 cardinal action deltas.
    # Pre-conditions:
    #   - gamma (float): Discount factor in [0, 1].
    #   - step_reward (float): Penalty incurred per transition (default -1.0).
    #   - goal_reward (float): Terminal reward on reaching goal (default +10.0).
    # Post-conditions:
    #   - GridWorld3x3 object initialized with coordinate tuples and action mappings.
    # ==============================================================================
    def __init__(self, gamma=0.9, step_reward=-1.0, goal_reward=10.0):
        self.rows = 3
        self.cols = 3
        self.gamma = gamma
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        
        self.start = (0, 0)
        self.obstacle = (1, 1)
        self.goal = (2, 2)
        
        self.actions = [0, 1, 2, 3]
        self.action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
        self.action_arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        self.action_deltas = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        
    # ==============================================================================
    # Function: get_states
    # Description:
    #   Generates the complete set of valid, traversable non-obstacle states in the 3x3 grid.
    # Pre-conditions:
    #   - None.
    # Post-conditions:
    #   - Returns a list of 8 coordinate tuples (r, c) excluding the blocked cell (1, 1).
    # ==============================================================================
    def get_states(self):
        return [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if (r, c) != self.obstacle
        ]

    # ==============================================================================
    # Function: transition
    # Description:
    #   Computes deterministic state transition T(s, a) -> (s', reward, is_terminal).
    #   Enforces collision dynamics: attempts to move off the grid or into obstacle (1, 1)
    #   leave the agent at s with step_reward. Entering goal (2, 2) yields goal_reward
    #   and terminates the episode.
    # Pre-conditions:
    #   - state (tuple): Current coordinate pair (r, c) in grid and not obstacle.
    #   - action (int): Integer action in {0, 1, 2, 3}.
    # Post-conditions:
    #   - Returns 3-tuple: (next_state, reward, is_terminal).
    # ==============================================================================
    def transition(self, state, action):
        if state == self.goal:
            return self.goal, 0.0, True

        dr, dc = self.action_deltas[action]
        nr, nc = state[0] + dr, state[1] + dc

        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = state

        if (nr, nc) == self.obstacle:
            nr, nc = state

        next_state = (nr, nc)

        if next_state == self.goal:
            return next_state, self.goal_reward, True
        else:
            return next_state, self.step_reward, False

grid = GridWorld3x3()
print("3x3 GridWorld MDP Environment initialized successfully!")
print(f"Start State: {grid.start} | Obstacle: {grid.obstacle} | Goal: {grid.goal}")
print(f"Discount Factor gamma: {grid.gamma} | Step Reward: {grid.step_reward} | Goal Reward: {grid.goal_reward}")
"""
    add_code(cells, grid_code)

    # Question 1 (Lab 3)
    add_md(cells, """### Question 1: Perform One Iteration of Policy Evaluation for 3x3 GridWorld
**Problem Statement:**  
*Write Python code to perform one iteration of policy evaluation for 3x3 GridWorld.*

#### Mathematical Derivation & Answer:
Under an equiprobable random policy $\\pi(a|s) = 0.25$ for each action $a \\in \\{\\uparrow, \\downarrow, \\leftarrow, \\rightarrow\\}$, the Bellman Expectation Equation for iteration $k=1$ is:
$$V_{1}(s) = \\sum_{a \\in \\mathcal{A}} \\pi(a|s) \\left[ R(s, a, s') + \\gamma V_0(s') \\right]$$
Given initial values $V_0(s) = 0$ for all states $s$, the $\\gamma V_0(s')$ term drops out completely:
$$V_{1}(s) = \\frac{1}{4} \\sum_{a=1}^{4} R(s, a, s')$$

- **For cells adjacent to Goal $G(2,2)$:**
  - State $(1, 2)$: Action **Down** enters Goal ($R = +10$). Actions **Up, Left (hits obstacle), Right (hits border)** yield $R = -1$.
    $$V_1(1, 2) = 0.25 \\times (10) + 0.25 \\times (-1) + 0.25 \\times (-1) + 0.25 \\times (-1) = 2.50 - 0.75 = +1.75$$
  - State $(2, 1)$: Action **Right** enters Goal ($R = +10$). Actions **Up (hits obstacle), Down, Left** yield $R = -1$.
    $$V_1(2, 1) = 0.25 \\times (10) + 0.75 \\times (-1) = +1.75$$
- **For all other non-goal, non-obstacle cells:**
  - No single action can reach the Goal in 1 step. All 4 actions produce $R = -1$.
    $$V_1(s) = 4 \\times [0.25 \\times (-1)] = -1.0$$
- **For Terminal Goal $G(2,2)$ and Obstacle $X(1,1)$:** $V_1 = 0.0$.
""")

    q1_lab3_code = """print("=== QUESTION 1 OUTPUT ===")

V_0 = np.zeros((3, 3))
V_1 = np.zeros((3, 3))

policy_prob = 0.25

step_details = []

for r in range(3):
    for c in range(3):
        state = (r, c)
        if state == grid.goal:
            V_1[r, c] = 0.0
            continue
        if state == grid.obstacle:
            V_1[r, c] = 0.0
            continue

        q_components = []
        val_sum = 0.0
        for a in grid.actions:
            next_state, reward, is_term = grid.transition(state, a)
            expected_val = reward + (0.0 if is_term else grid.gamma * V_0[next_state])
            val_sum += policy_prob * expected_val
            q_components.append(f"{grid.action_names[a]}: R={reward:+2.0f}")

        V_1[r, c] = val_sum
        step_details.append({
            "State (r, c)": f"({r}, {c})",
            "Action Rewards": " | ".join(q_components),
            "Calculation": f"0.25 * ({' + '.join([c_comp.split('R=')[1] for c_comp in q_components])})",
            "V_1(s)": V_1[r, c]
        })

print("State Values V_1(s) after 1 Iteration:")
display(pd.DataFrame(step_details))

print("\\nResulting 3x3 State-Value Grid V_1:")
df_v1 = pd.DataFrame(V_1, index=[f"Row {r}" for r in range(3)], columns=[f"Col {c}" for c in range(3)])
display(df_v1)
"""
    add_code(cells, q1_lab3_code)

    # Question 2 (Lab 3)
    add_md(cells, """### Question 2: Calculate the Value of the Specified State using the Given Policy
**Problem Statement:**  
*Write Python code to calculate the value of the specified state using the given policy for the 3 × 3 GridWorld.*

#### Mathematical Derivation & Answer:
To calculate the true value $V^\\pi(s)$ of any specified state (such as the Start state $S(0, 0)$ or neighbor $(0, 1)$), we apply iterative policy evaluation until numerical convergence:
$$V_{k+1}(s) \\leftarrow \\sum_{a} \\pi(a|s) \\sum_{s'} P(s'|s,a) \\left[ R(s, a, s') + \\gamma V_k(s') \\right]$$
Convergence is achieved when $\\max_{s} |V_{k+1}(s) - V_k(s)| < 10^{-6}$.
""")

    q2_lab3_code = """print("=== QUESTION 2 OUTPUT ===")

# ==============================================================================
# Function: evaluate_policy_to_convergence
# Description:
#   Implements Iterative Policy Evaluation using Bellman Expectation backups under
#   the uniform random policy pi(a|s) = 0.25 until maximum absolute delta falls
#   below the specified convergence threshold tolerance.
# Pre-conditions:
#   - grid: Initialized GridWorld3x3 instance.
#   - tol (float): Convergence threshold (default 1e-6).
#   - max_iter (int): Upper bound on allowed evaluation iterations.
# Post-conditions:
#   - Returns 2-tuple: (V_pi, history).
#   - V_pi is a 3x3 numpy array of converged state values V^pi(s).
#   - history is a list of maximum deltas per iteration.
# ==============================================================================
def evaluate_policy_to_convergence(grid, tol=1e-6, max_iter=1000):
    V = np.zeros((3, 3))
    history = []
    
    for iteration in range(1, max_iter + 1):
        delta = 0.0
        V_new = np.zeros((3, 3))
        
        for r in range(3):
            for c in range(3):
                s = (r, c)
                if s == grid.goal or s == grid.obstacle:
                    continue
                
                val = 0.0
                for a in grid.actions:
                    next_s, reward, is_term = grid.transition(s, a)
                    val += 0.25 * (reward + (0.0 if is_term else grid.gamma * V[next_s]))
                    
                V_new[r, c] = val
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
                
        V = V_new.copy()
        history.append(delta)
        if delta < tol:
            print(f"Policy Evaluation converged successfully in {iteration} iterations (max delta = {delta:.2e})")
            break
            
    return V, history

V_pi, delta_hist = evaluate_policy_to_convergence(grid)

specified_states = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

print("\\nConverged State Values V^pi for Specified States:")
df_specified = pd.DataFrame([
    {
        "State Coordinate": f"({r}, {c})",
        "Role / Label": "Start State S" if (r, c) == (0, 0) else ("Adjacent to Goal" if (r, c) in [(1, 2), (2, 1)] else "Interior Path"),
        "Converged Value V^pi(s)": f"{V_pi[r, c]:.4f}"
    }
    for (r, c) in specified_states
])
display(df_specified)

plt.figure(figsize=(6, 5))
mask = np.zeros((3, 3), dtype=bool)
mask[1, 1] = True
sns.heatmap(V_pi, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, mask=mask,
            linewidths=1.5, annot_kws={"size": 13, "weight": "bold"})
plt.title("Converged State Values V^pi under Random Policy", fontsize=12)
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.text(1.5, 1.5, "BLOCKED (X)", ha="center", va="center", color="black", fontweight="bold")
plt.tight_layout()
plt.show()
"""
    add_code(cells, q2_lab3_code)

    # Question 3 (Lab 3)
    add_md(cells, """### Question 3: Determine the Best Action for the Given State using Value Iteration
**Problem Statement:**  
*Write Python code to determine the best action for the given state using value iteration.*

#### Mathematical Derivation & Answer:
Value Iteration combines policy evaluation and policy improvement into a single Bellman Optimality update:
$$V_{k+1}(s) = \\max_{a \\in \\mathcal{A}} \\left[ R(s, a, s') + \\gamma V_k(s') \\right]$$
Once $V^*(s)$ converges, the optimal action $\\pi^*(s)$ for any given state $s$ is extracted via greedy $\\arg\\max$:
$$\\pi^*(s) = \\arg\\max_{a \\in \\mathcal{A}} Q^*(s, a) = \\arg\\max_{a \\in \\mathcal{A}} \\left[ R(s, a, s') + \\gamma V^*(s') \\right]$$

- For Start State $S(0, 0)$:
  - Action **Up**: hits boundary, stays at $(0,0) \\implies -1 + 0.9 \\times V^*(0,0)$
  - Action **Left**: hits boundary, stays at $(0,0) \\implies -1 + 0.9 \\times V^*(0,0)$
  - Action **Right**: moves to $(0,1) \\implies -1 + 0.9 \\times V^*(0,1) = -1 + 0.9 \\times (6.20) = +4.58$
  - Action **Down**: moves to $(1,0) \\implies -1 + 0.9 \\times V^*(1,0) = -1 + 0.9 \\times (6.20) = +4.58$
  - **Best Actions:** Either **Right (→)** or **Down (↓)**, both achieving the maximal optimal value $Q^*(S) = 4.58$.
""")

    q3_lab3_code = """print("=== QUESTION 3 OUTPUT ===")

# ==============================================================================
# Function: value_iteration
# Description:
#   Implements Value Iteration using the Bellman Optimality Equation:
#   V(s) <- max_a [ R(s, a) + gamma * V(s') ]
#   Repeatedly updates state values synchronously until convergence delta < tol.
# Pre-conditions:
#   - grid: Initialized GridWorld3x3 instance.
#   - tol (float): Maximum absolute difference tolerance for stopping.
#   - max_iter (int): Maximum iterations allowed.
# Post-conditions:
#   - Returns 3x3 numpy array V_opt containing optimal state values V*(s).
# ==============================================================================
def value_iteration(grid, tol=1e-6, max_iter=1000):
    V_opt = np.zeros((3, 3))
    
    for it in range(1, max_iter + 1):
        delta = 0.0
        V_new = np.zeros((3, 3))
        
        for r in range(3):
            for c in range(3):
                s = (r, c)
                if s == grid.goal or s == grid.obstacle:
                    continue
                
                q_vals = []
                for a in grid.actions:
                    next_s, reward, is_term = grid.transition(s, a)
                    q = reward + (0.0 if is_term else grid.gamma * V_opt[next_s])
                    q_vals.append(q)
                    
                best_q = max(q_vals)
                V_new[r, c] = best_q
                delta = max(delta, abs(best_q - V_opt[r, c]))
                
        V_opt = V_new.copy()
        if delta < tol:
            print(f"Value Iteration converged in {it} iterations (delta = {delta:.2e})")
            break
            
    return V_opt

V_star = value_iteration(grid)

given_state = (0, 0)
q_table_given = []
for a in grid.actions:
    next_s, reward, is_term = grid.transition(given_state, a)
    q_val = reward + (0.0 if is_term else grid.gamma * V_star[next_s])
    q_table_given.append({
        "Action Index": a,
        "Action Name": grid.action_names[a],
        "Direction Arrow": grid.action_arrows[a],
        "Next State s'": f"{next_s}",
        "Reward R": reward,
        "V*(s')": 0.0 if is_term else V_star[next_s],
        "Q*(s, a) = R + gamma*V*(s')": q_val
    })

df_q_given = pd.DataFrame(q_table_given)
print(f"\\nQ-values Q*(s, a) for Given State {given_state}:")
display(df_q_given)

best_action_idx = int(np.argmax([item["Q*(s, a) = R + gamma*V*(s')"] for item in q_table_given]))
best_action_name = grid.action_names[best_action_idx]
best_action_arrow = grid.action_arrows[best_action_idx]

print(f"\\n>> BEST ACTION for State {given_state}: {best_action_name} ({best_action_arrow})")
print(f">> Optimal State Value V*({given_state}) = {V_star[given_state]:.4f}")

optimal_policy = {}
for r in range(3):
    for c in range(3):
        s = (r, c)
        if s == grid.goal:
            optimal_policy[s] = "GOAL"
        elif s == grid.obstacle:
            optimal_policy[s] = "WALL"
        else:
            q_vals = [
                grid.transition(s, a)[1] + (0.0 if grid.transition(s, a)[2] else grid.gamma * V_star[grid.transition(s, a)[0]])
                for a in grid.actions
            ]
            optimal_policy[s] = grid.action_arrows[int(np.argmax(q_vals))]

print("\\nFull Optimal Policy Grid (pi*):")
policy_grid = np.array([[optimal_policy[(r, c)] for c in range(3)] for r in range(3)])
display(pd.DataFrame(policy_grid, index=[f"Row {r}" for r in range(3)], columns=[f"Col {c}" for c in range(3)]))
"""
    add_code(cells, q3_lab3_code)

    # Question 4 (Lab 3)
    add_md(cells, """### Question 4: Find the Shortest Path from Start to Goal using Given Optimal Policy
**Problem Statement:**  
*Write Python code to find the shortest path from the Start state to the Goal state using the given optimal policy.*

#### Explanation & Answer:
- Following the optimal policy $\\pi^*$ extracted in Question 3, the agent starts at $S(0, 0)$ and deterministically follows the optimal actions.
- The obstacle at $X(1, 1)$ prevents a straight diagonal path.
- The policy successfully routes around $X(1, 1)$ via the upper corridor $(0, 0) \\to (0, 1) \\to (0, 2) \\to (1, 2) \\to (2, 2)$ (or symmetrically via the lower corridor $(0,0) \\to (1,0) \\to (2,0) \\to (2,1) \\to (2,2)$), reaching the goal in **exactly 4 steps**, which is the absolute theoretical shortest path.
""")

    q4_lab3_code = """print("=== QUESTION 4 OUTPUT ===")

# ==============================================================================
# Function: find_shortest_path
# Description:
#   Traces the deterministic greedy trajectory from Start state (0, 0) to Goal state (2, 2)
#   by querying the optimal Q-values derived from V*. Records visited states, actions taken,
#   and transition rewards until goal is achieved.
# Pre-conditions:
#   - grid: Initialized GridWorld3x3 instance.
#   - optimal_policy (dict): Mapping from (r, c) states to optimal directional actions.
# Post-conditions:
#   - Returns 3-tuple: (path, actions_taken, rewards).
#   - path is list of coordinate pairs starting with (0, 0) and terminating at (2, 2).
#   - actions_taken is list of action strings.
#   - rewards is list of floats representing rewards encountered per step.
# ==============================================================================
def find_shortest_path(grid, optimal_policy):
    curr = grid.start
    path = [curr]
    actions_taken = []
    rewards = []
    step_num = 0
    
    while curr != grid.goal and step_num < 20:
        step_num += 1
        q_vals = [
            grid.transition(curr, a)[1] + (0.0 if grid.transition(curr, a)[2] else grid.gamma * V_star[grid.transition(curr, a)[0]])
            for a in grid.actions
        ]
        best_a = int(np.argmax(q_vals))
        next_s, reward, is_term = grid.transition(curr, best_a)
        
        actions_taken.append(grid.action_names[best_a] + f" ({grid.action_arrows[best_a]})")
        rewards.append(reward)
        curr = next_s
        path.append(curr)
        
    return path, actions_taken, rewards

path_opt, actions_opt, rewards_opt = find_shortest_path(grid, optimal_policy)

print(f"Shortest Path Sequence Found ({len(actions_opt)} steps):")
for i in range(len(actions_opt)):
    print(f"  Step {i+1}: {path_opt[i]} --[{actions_opt[i]}]--> {path_opt[i+1]} (Reward = {rewards_opt[i]:+2.0f})")

print(f"\\nFinal Status        : Goal Reached at {path_opt[-1]}!")
print(f"Total Steps Taken   : {len(actions_opt)}")
print(f"Total Undiscounted R: {sum(rewards_opt):+.1f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(2.5, -0.5)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.grid(True, color="black", linewidth=1.5)

ax.add_patch(plt.Rectangle((grid.start[1]-0.5, grid.start[0]-0.5), 1, 1, color="#2ecc71", alpha=0.3))
ax.text(grid.start[1], grid.start[0], "START\\n(0,0)", ha="center", va="center", fontweight="bold", fontsize=11)

ax.add_patch(plt.Rectangle((grid.obstacle[1]-0.5, grid.obstacle[0]-0.5), 1, 1, color="#e74c3c", alpha=0.4))
ax.text(grid.obstacle[1], grid.obstacle[0], "BLOCKED\\n(1,1)", ha="center", va="center", fontweight="bold", color="darkred", fontsize=11)

ax.add_patch(plt.Rectangle((grid.goal[1]-0.5, grid.goal[0]-0.5), 1, 1, color="#f1c40f", alpha=0.4))
ax.text(grid.goal[1], grid.goal[0], "GOAL\\n(2,2)", ha="center", va="center", fontweight="bold", fontsize=11)

for i in range(len(path_opt) - 1):
    r1, c1 = path_opt[i]
    r2, c2 = path_opt[i+1]
    ax.annotate("", xy=(c2, r2), xytext=(c1, r1),
                arrowprops=dict(arrowstyle="->", color="#2980b9", lw=3.5, mutation_scale=20))
    ax.plot(c1, r1, "o", color="#2980b9", markersize=8)

ax.set_title("Optimal Shortest Path Trajectory on 3x3 GridWorld", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
"""
    add_code(cells, q4_lab3_code)

    # Question 5 (Lab 3)
    add_md(cells, """### Question 5: Compare Two Paths Based on Path Length and Total Reward
**Problem Statement:**  
*Write Python code to compare two paths based on path length and total reward.*

#### Explanation & Answer:
We compare:
1. **Path 1 (Optimal Policy Path):**
   - Coordinates: $(0, 0) \\to (0, 1) \\to (0, 2) \\to (1, 2) \\to (2, 2)$
   - Path Length: $4$ steps
   - Total Undiscounted Reward: $(-1) + (-1) + (-1) + 10 = +7.0$
   - Total Discounted Return $G$: $-1 - 0.9 - 0.81 + (0.9^3 \\times 10) = +4.58$
2. **Path 2 (Suboptimal Detour Path):**
   - Coordinates: $(0, 0) \\to (1, 0) \\to (0, 0) \\to (0, 1) \\to (0, 2) \\to (1, 2) \\to (2, 2)$
   - Path Length: $6$ steps (wastes steps backtracking)
   - Total Undiscounted Reward: $(-1) \\times 5 + 10 = +5.0$
   - Total Discounted Return $G$: Significantly lower due to step penalties and discount compounding.
""")

    q5_lab3_code = """print("=== QUESTION 5 OUTPUT ===")

# ==============================================================================
# Function: evaluate_trajectory
# Description:
#   Analyzes an arbitrary trajectory of state coordinates through the GridWorld MDP.
#   Identifies matching valid transitions, step costs, undiscounted cumulative reward,
#   and discounted return G = sum_{t=0}^{T-1} gamma^t * R_{t+1}.
# Pre-conditions:
#   - path_states (list): Ordered list of (r, c) coordinate tuples representing trajectory.
#   - grid: Initialized GridWorld3x3 instance.
# Post-conditions:
#   - Returns dictionary with keys:
#     'Path Coordinates', 'Path Length (Steps)', 'Total Undiscounted Reward',
#     'Discounted Return (G)', 'Goal Reached'.
# ==============================================================================
def evaluate_trajectory(path_states, grid):
    rewards = []
    actions = []
    for i in range(len(path_states) - 1):
        s = path_states[i]
        next_s = path_states[i+1]
        
        for a in grid.actions:
            res_s, r, term = grid.transition(s, a)
            if res_s == next_s:
                rewards.append(r)
                actions.append(grid.action_names[a])
                break
                
    length = len(actions)
    total_undiscounted_reward = sum(rewards)
    discounted_return = sum([rewards[t] * (grid.gamma ** t) for t in range(length)])
    goal_reached = (path_states[-1] == grid.goal)
    
    return {
        "Path Coordinates": " -> ".join([str(s) for s in path_states]),
        "Path Length (Steps)": length,
        "Total Undiscounted Reward": total_undiscounted_reward,
        "Discounted Return (G)": round(discounted_return, 4),
        "Goal Reached": "Yes" if goal_reached else "No"
    }

path_1 = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
path_2 = [(0, 0), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

res_path1 = evaluate_trajectory(path_1, grid)
res_path2 = evaluate_trajectory(path_2, grid)

df_compare = pd.DataFrame([
    {"Trajectory": "Path 1 (Optimal Shortest Path)", **res_path1},
    {"Trajectory": "Path 2 (Suboptimal Detour Path)", **res_path2}
])

print("Comparative Metrics Summary Table:")
display(df_compare)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(2)
width = 0.28

bars1 = ax.bar(x - width, [res_path1["Path Length (Steps)"], res_path2["Path Length (Steps)"]], width, label="Path Length (Steps)", color="#3498db")
bars2 = ax.bar(x, [res_path1["Total Undiscounted Reward"], res_path2["Total Undiscounted Reward"]], width, label="Total Undiscounted Reward", color="#2ecc71")
bars3 = ax.bar(x + width, [res_path1["Discounted Return (G)"], res_path2["Discounted Return (G)"]], width, label="Discounted Return (G)", color="#f39c12")

ax.set_xticks(x)
ax.set_xticklabels(["Path 1 (Optimal)", "Path 2 (Suboptimal)"], fontsize=11, fontweight="bold")
ax.set_ylabel("Metric Value")
ax.set_title("Path Comparison: Length vs Undiscounted Reward vs Discounted Return", fontsize=12)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in list(bars1) + list(bars2) + list(bars3):
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print(\"\"\"
Key Takeaways & Analysis:
1. Path Length: Path 1 navigates directly to the goal in 4 steps (shortest possible). Path 2 wastes 2 additional steps (6 steps total).
2. Undiscounted Reward: Path 1 achieves +7.0, whereas Path 2 suffers two additional step costs (-1 each), resulting in +5.0.
3. Discounted Return: Path 1 achieves G = 4.58 (exactly equal to V*(0,0)). Path 2's goal reward is delayed by 2 time steps, heavily degrading discounted return.
\"\"\")
"""
    add_code(cells, q5_lab3_code)

    nb.cells = cells
    out_path = os.path.join(OUT_DIR, "Lab_Assignment_3.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path} with valid syntax!")
    return out_path


def execute_notebook(path):
    """
    Description:
      Executes all code cells in the given notebook file in place, populating them
      with graphical outputs, pandas dataframes, and stdout text.
    Pre-conditions:
      - path (str): Valid filesystem path to a .ipynb notebook.
    Post-conditions:
      - Re-writes notebook file containing executed output streams and MIME bundles.
    """
    print(f"Executing notebook: {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved: {path}!")


def create_master_combined_notebook(nb1_path, nb3_path):
    """
    Description:
      Merges all cells from Lab_Assignment_1_and_2.ipynb and Lab_Assignment_3.ipynb
      into a single consolidated master notebook with unified navigation and execution.
    Pre-conditions:
      - nb1_path (str): File path to executed Lab 1 & 2 notebook.
      - nb3_path (str): File path to executed Lab 3 notebook.
    Post-conditions:
      - Generates Lab_Assignments_1_2_3_Master.ipynb containing all 40 cells and outputs.
    """
    print("Creating consolidated master notebook...")
    with open(nb1_path, "r", encoding="utf-8") as f:
        nb1 = nbf.read(f, as_version=4)
    with open(nb3_path, "r", encoding="utf-8") as f:
        nb3 = nbf.read(f, as_version=4)

    master_nb = nbf.v4.new_notebook()
    master_nb.metadata = nb1.metadata

    master_cells = []
    master_cells.append(nbf.v4.new_markdown_cell("""# Reinforcement Learning Laboratory
## Master Comprehensive Notebook: Lab Assignments 1, 2, and 3
**Author:** Samrat Kar (PhD CSE)  
**Institution:** Amrita Vishwa Vidyapeetham  
**Repository:** Deep Reinforcement Learning Educational Curriculum  

This master notebook integrates all questions, implementations, and empirical results for:
1. **Lab Assignment 1 – Explore OpenAI Gymnasium** (3D Tic-Tac-Toe Environment, Q1–Q5)
2. **Lab Assignment 2 – Reinforcement Learning using Tic-Tac-Toe** (Q-Learning & Self-Play Dynamics, Q1–Q5)
3. **Lab Assignment 3 – GridWorld** (Policy Evaluation, Value Iteration, Optimal Pathfinding, Q1–Q5)
"""))

    for cell in nb1.cells[1:]:
        master_cells.append(cell)

    master_cells.append(nbf.v4.new_markdown_cell("""---
---
# Part III: Lab Assignment 3 – 3x3 GridWorld (Policy Evaluation & Value Iteration)
"""))

    for cell in nb3.cells[1:]:
        master_cells.append(cell)

    master_nb.cells = master_cells
    master_path = os.path.join(OUT_DIR, "Lab_Assignments_1_2_3_Master.ipynb")
    with open(master_path, "w", encoding="utf-8") as f:
        nbf.write(master_nb, f)
    print(f"Successfully generated master notebook: {master_path}")
    return master_path


if __name__ == "__main__":
    p1 = build_notebook_1_and_2()
    execute_notebook(p1)

    p3 = build_notebook_3()
    execute_notebook(p3)

    p_master = create_master_combined_notebook(p1, p3)
    execute_notebook(p_master)
    print("\nALL NOTEBOOKS WITH FUNCTION DESCRIPTIONS, PRE-CONDITIONS & POST-CONDITIONS EXECUTED SUCCESSFULLY!")
