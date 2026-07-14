---
layout: post
---

# Solutions to MCQs (Lecture 8)

1. **c) To break the temporal correlations in sequential data, making it pseudo-i.i.d.** (Sequential states $S_t$ and $S_{t+1}$ are highly correlated; random sampling from a buffer destroys this correlation, stabilizing SGD).
2. **a) By using a separate Target Network that is only updated periodically.** (The Target Network's weights $\theta'$ are held constant while the Online Network $\theta$ is trained, providing a fixed target).
3. **b) Due to the $\max$ operator applied to noisy Q-value estimates.** (The expected maximum of noisy estimates is strictly greater than the true maximum).
4. **c) By using the Online Network to select the action and the Target Network to evaluate it.** (Decoupling selection from evaluation).
5. **b) The Target Network (which evaluates the action) is unlikely to have the exact same overestimation error for that action.** (Because $\theta'$ is independent from $\theta$ in the short term, its noise is uncorrelated).
