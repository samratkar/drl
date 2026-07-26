---
layout: post
---

# Numerical Solutions (Lecture 12: Implementation Frameworks)

1. **Trial Step Count Calculation:**
   * **Total steps per Session:**
     $$ 500 \text{ episodes} \times 200 \text{ steps/episode} = 100,000 \text{ steps} $$
   * **Total steps across Trial (5 Sessions):**
     $$ 100,000 \text{ steps/session} \times 5 \text{ sessions} = 500,000 \text{ steps} $$
   * The total number of environment step interactions executed across the entire Trial is $500,000$.

2. **DQN Target Value Calculation:**
   * Because the next state $S'$ is terminal (`terminated = True`), there are no future discounted rewards from $S'$.
   * Therefore, the target Q-value $y$ is simply the immediate reward $R$:
     $$ y = R = -1.0 $$
   * Note: The predictions $Q(S', a)$ and $Q^-(S', a)$ are ignored for terminal states in the Bellman backup calculation:
     $$ y_i = R_i + (1 - d_i) \gamma \max_{a'} Q^-(S'_i, a') $$
     where $d_i = 1$ when terminal, zeroing out the bootstrap term.
