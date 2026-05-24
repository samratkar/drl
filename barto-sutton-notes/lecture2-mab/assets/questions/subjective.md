---
layout: post
---

# Subjective Questions - Lecture 2 (Multi-Armed Bandits)

1. **Exploration-Exploitation Trade-off:** Explain the conflict between exploration and exploitation in the context of the k-armed bandit problem. Why can't we just do one or the other?

2. **Sample-Average vs. Constant Step-size:** Derive the incremental update rule for sample-average estimates. Explain why constant step-size is preferred for non-stationary problems.

3. **Optimistic Initial Values:** How does setting high initial values for $Q(a)$ force exploration? What are the limitations of this method?

4. **UCB Action Selection:** Explain the intuition behind the UCB formula. How does it balance the current value estimate with the uncertainty of that estimate?

5. **Gradient Bandit Algorithm:** Describe how the gradient bandit algorithm works without estimating action values. Why is the baseline $\bar{R}_t$ important?

6. **The 10-Armed Testbed:** In Figure 2.2, why does $\epsilon=0.1$ perform better than $\epsilon=0$ in the long run, even though it learns more slowly initially?

7. **Convergence Conditions:** What are the two mathematical conditions for step-size convergence? Why does constant $\alpha$ intentionally violate one of them?

8. **Non-stationarity:** Give a real-world example of a non-stationary bandit problem. Why would a sample-average method fail in this scenario?
