---
layout: post
---

# Programming Questions (Lecture 11: Combined, Advanced PG & Model-Based RL)

1. **Implementing MCTS Selection & Expansion:**
   Implement the selection and expansion steps for the `MCTSNode` class below. Your selection step should recursively call child selection based on the highest UCT score, and the expansion step should instantiate a new child node using one of the legal actions.

   ```python
   import math

   class Node:
       def __init__(self, state, parent=None, action=None):
           self.state = state
           self.parent = parent
           self.action = action
           self.children = []
           self.visit_count = 0
           self.total_value = 0
           self.unexpanded_actions = state.get_legal_actions()

       def is_fully_expanded(self):
           return len(self.unexpanded_actions) == 0

       def q_value(self):
           if self.visit_count == 0:
               return 0.0
           return self.total_value / self.visit_count

   def uct_select(node, c=1.414):
       # TODO: Implement selection using UCT formula
       pass

   def expand(node):
       # TODO: Implement expansion of a node
       pass
   ```

2. **Implementing DAgger Dataset Aggregation Loop:**
   Implement the dataset aggregation loop for the `train_dagger_epoch` function below. In this function, the agent policy rolls out in the environment. At each state visited by the agent, you must query the expert policy to get the correct action label, and add the state-expert_action pair to the dataset `D`.

   ```python
   def train_dagger_epoch(env, agent_policy, expert_policy, D):
       """
       Args:
           env: The Gym environment.
           agent_policy: The policy network we are training.
           expert_policy: The oracle expert policy to query.
           D: List of tuples (state, action) representing the aggregated dataset.
       """
       state, _ = env.reset()
       terminated = False
       truncated = False
       
       while not (terminated or truncated):
           # 1. Select action using current agent policy
           # action = agent_policy.select_action(state)
           # 2. Step environment to get next state
           # next_state, reward, terminated, truncated, _ = env.step(action)
           # 3. Query expert to get action label for the state visited by the agent
           # expert_action = expert_policy.select_action(state)
           # 4. Add the (state, expert_action) pair to dataset D
           # TODO: Implement the above steps
           pass
   ```
