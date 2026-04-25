1. environment - trees shift - env change - **state** changes 
2. agent - Sara
3. agent is interacting with environment 
4. episode - numner of steps you take to predict the overall return
5. policy - state to action mapping
6. $\pi$ - policy function = $\pi(a|s)$ - probability of taking action a given state s
7. $P(s'|s,a)$ - transition probability of going to state s' given state s and action a
8. $G_t$ - return at time t : sum of discounted rewards from time t to the end of the episode : $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$ where $\gamma$ is the discount factor
9. $R_{t+1}$ - reward at time t+1 : immediate reward : 