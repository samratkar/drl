# Numerical Questions (Chapter 9: Function Approximation)

**Question 1: Linear Semi-gradient TD Update**
Assume a linear function approximator $\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s)$. 
At time step $t$, the agent is in state $S_t$ with feature vector $\mathbf{x}(S_t) = [1, 2, 0]^T$. 
The current weight vector is $\mathbf{w}_t = [0.5, -0.1, 1.0]^T$. 
The agent takes an action, receives a reward $R_{t+1} = 10$, and transitions to state $S_{t+1}$ with feature vector $\mathbf{x}(S_{t+1}) = [1, 0, 1]^T$.
Given a discount factor $\gamma = 0.9$ and learning rate $\alpha = 0.1$:
1. Calculate the estimated value of the current state, $\hat{v}(S_t, \mathbf{w}_t)$.
2. Calculate the estimated value of the next state, $\hat{v}(S_{t+1}, \mathbf{w}_t)$.
3. Calculate the TD error, $\delta_t$.
4. Calculate the new weight vector, $\mathbf{w}_{t+1}$.

**Question 2: Tile Coding Step-Size**
You are using a tile coding system to approximate the value function of a robotic arm. The system uses 8 overlapping tilings. You want the algorithm to move exactly one-quarter (1/4) of the way toward the target upon every update to ensure stability. 
What should the learning rate $\alpha$ be set to?
