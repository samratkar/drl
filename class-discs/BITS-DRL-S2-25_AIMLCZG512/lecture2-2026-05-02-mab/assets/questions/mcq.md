# Chapter 2: Multi-Armed Bandits — Multiple Choice Questions

*Based on Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.), Chapter 2*

---

### Q1. Evaluative vs Instructive Feedback
What distinguishes evaluative feedback (as in RL) from instructive feedback (as in supervised learning)?

(a) Evaluative feedback tells the agent the best action to take  
(b) Evaluative feedback indicates how good the taken action was, but not whether it was the best or worst  
(c) Evaluative feedback always provides the gradient of the loss function  
(d) Evaluative feedback requires a labeled dataset  

---

### Q2. k-Armed Bandit Definition
In the k-armed bandit problem, what does the agent try to maximize?

(a) The number of unique arms pulled  
(b) The total expected reward over a time period  
(c) The variance of observed rewards  
(d) The number of times the best arm is identified  

---

### Q3. True Value vs Estimated Value
$q_*(a)$ represents the true value of action $a$. What is $Q_t(a)$?

(a) The true value of action $a$ at time $t$  
(b) The agent's estimate of $q_*(a)$ at time $t$, based on observed rewards  
(c) The maximum possible reward from action $a$  
(d) The reward received at time $t$ from action $a$  

---

### Q4. Greedy Action
The greedy action at time $t$ is defined as $A_t = \argmax_a Q_t(a)$. What is the main weakness of always selecting the greedy action?

(a) It requires too much computation  
(b) It always selects the worst arm  
(c) It may lock onto a suboptimal arm and never discover better alternatives  
(d) It requires knowledge of $q_*(a)$  

---

### Q5. Epsilon-Greedy Exploration
In an $\epsilon$-greedy method with $\epsilon = 0.1$ and $k = 10$ arms, what is the probability of selecting any specific non-greedy arm on a given step?

(a) $0.1$  
(b) $0.01$  
(c) $0.9$  
(d) $0.1 / 10 = 0.01$  

---

### Q6. When to Increase Exploration
Under which conditions should $\epsilon$ be larger (more exploration)?

(a) When the rewards have low variance and $q_*(a)$ is fixed  
(b) When the rewards have high variance or $q_*(a)$ changes over time  
(c) When the agent has already found the optimal arm  
(d) When $k$ is very small (e.g., $k = 2$)  

---

### Q7. Sample-Average Method
The sample-average method estimates $Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}$. What happens to this estimate as the number of selections of arm $a$ goes to infinity?

(a) It diverges  
(b) It converges to $q_*(a)$ by the law of large numbers  
(c) It oscillates between 0 and $q_*(a)$  
(d) It converges to 0  

---

### Q8. Incremental Update Rule
The incremental update $Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]$ is equivalent to which operation?

(a) Exponential moving average of all rewards  
(b) Simple average of the $n$ most recent rewards  
(c) Simple average of all $n$ observed rewards for that arm  
(d) Weighted average favoring recent rewards  

---

### Q9. Constant Step-Size Weighting
With a constant step-size $\alpha = 0.1$, the weight on reward $R_i$ (received $n - i$ steps ago) is $\alpha(1-\alpha)^{n-i}$. Which reward receives the most weight?

(a) The oldest reward $R_1$  
(b) The most recent reward $R_n$  
(c) All rewards receive equal weight  
(d) The median reward  

---

### Q10. Stationary vs Non-Stationary
Why is the sample-average method ($\alpha_n = 1/n$) unsuitable for non-stationary problems?

(a) It gives too much weight to recent rewards  
(b) It gives equal weight to all past rewards, including stale ones from a distribution that no longer applies  
(c) It doesn't converge  
(d) It requires knowing $q_*(a)$ in advance  

---

### Q11. Convergence Conditions
The convergence conditions for $Q_n \to q_*(a)$ are $\sum \alpha_n = \infty$ and $\sum \alpha_n^2 < \infty$. Which step-size satisfies both?

(a) $\alpha_n = \alpha$ (constant)  
(b) $\alpha_n = 1/n$  
(c) $\alpha_n = 1/n^2$  
(d) $\alpha_n = n$  

---

### Q12. Why Constant $\alpha$ Doesn't Converge
A constant step-size $\alpha$ violates $\sum \alpha_n^2 < \infty$. Why is this actually desirable for non-stationary problems?

(a) It causes the estimate to diverge, which is useful  
(b) It means the estimate never fully settles, allowing it to continuously track a changing reward distribution  
(c) It makes the algorithm faster to compute  
(d) It reduces variance of the estimates  

---

### Q13. Initial Bias — Sample Average
With sample-average ($\alpha_n = 1/n$) and $Q_1 = 5$, what is $Q_2$ after observing $R_1 = 1.0$?

(a) $Q_2 = 5.0$  
(b) $Q_2 = 3.0$  
(c) $Q_2 = 1.0$  
(d) $Q_2 = 4.6$  

---

### Q14. Initial Bias — Constant Alpha
With constant $\alpha = 0.1$ and $Q_1 = 5$, after 10 observations, the weight remaining on $Q_1$ is $(1-\alpha)^{10} = 0.9^{10} \approx 0.35$. What does this mean?

(a) $Q_1$ has been completely forgotten  
(b) $Q_1$ still contributes $5 \times 0.35 = 1.74$ to the current estimate  
(c) The estimate has converged to $q_*(a)$  
(d) The estimate equals $0.35$  

---

### Q15. Optimistic Initial Values — Mechanism
Why does setting $Q_1 = 5$ (optimistic) with greedy selection and constant $\alpha$ encourage exploration?

(a) The agent randomly selects arms  
(b) Every arm's initial estimate is above the true value, so every reward feels "disappointing," pushing the agent to try other arms  
(c) The high initial value increases the reward received  
(d) The agent knows which arm is best from the initial value  

---

### Q16. UCB — No Random Branch
How does UCB differ from $\epsilon$-greedy in its action selection structure?

(a) UCB has an exploit branch and a random explore branch, just like $\epsilon$-greedy  
(b) UCB uses a single argmax over a combined score (value + uncertainty bonus), with no random selection  
(c) UCB only explores and never exploits  
(d) UCB requires a neural network  

---

### Q17. UCB — Under-Explored Arms
In UCB, an arm with $N_t(a) = 0$ (never tried) has what score?

(a) Zero  
(b) Equal to $Q_t(a)$  
(c) Infinite (the bonus $c\sqrt{\ln t / N_t(a)} \to \infty$), guaranteeing it is tried first  
(d) Negative  

---

### Q18. Gradient Bandit — Preferences
In the gradient bandit algorithm, $H_t(a)$ represents the preference for action $a$. What happens if we add 100 to all preferences?

(a) The action probabilities change dramatically  
(b) The selected action changes  
(c) Nothing — only relative differences between preferences matter (softmax is shift-invariant)  
(d) The algorithm diverges  

---

### Q19. Gradient Bandit — Baseline Purpose
In the gradient bandit update $H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R}_t)(\mathbb{1}_{A_t=a} - \pi_t(a))$, what is the role of the baseline $\bar{R}_t$?

(a) It changes the expected gradient direction  
(b) It doesn't affect the expected gradient, but dramatically reduces variance by asking "was this reward better than average?" instead of "was this reward positive?"  
(c) It prevents the algorithm from converging  
(d) It replaces the step-size $\alpha$  

---

### Q20. Parameter Study — Overall Lesson
In the parameter study (Figure 2.6), every method shows an inverted-U curve. What does this shape mean?

(a) All methods perform identically  
(b) Too little exploration (left) misses the best arm; too much exploration (right) wastes time on bad arms; the peak is the sweet spot  
(c) The methods are not sensitive to their parameters  
(d) Larger parameters always produce better performance  
