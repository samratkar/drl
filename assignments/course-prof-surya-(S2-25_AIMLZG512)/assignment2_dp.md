### Course : (S2-25_AIMLZG512)
### Assignment : 2 : dynamic programming implementation 
#### Question : Implementing Policy Evaluation and Policy Improvement
#### Marks : 10
---

#### Question : 

There is a 3X3 grid world case study in the attached jupyter notebook. 
The details of the environment is already implemented. 
Go through the detailed documentation and code provided in the notebook on the context of the environment and the problem statement. Implement the following functions in the notebook to complete the assignment. - 
1. `compute_q_from_v(env, V, gamma=0.9)` : This function computes the state-action value function Q from the state value function V using the Bellman expectation equation for a given policy. - **MARKS : 2**
2. `policy_improvement(env, V, gamma=0.9, epsilon=0.1):` : This function computes the improved epsilon-soft policy from the state value function V using the computed Q-table from the previous function. - **MARKS : 4**
3. `policy_evaluation(env, policy, gamma=0.9, theta=1e-6)` : This function evaluates the given policy by computing the state value function V using iterative policy evaluation until convergence. - **MARKS : 4**


