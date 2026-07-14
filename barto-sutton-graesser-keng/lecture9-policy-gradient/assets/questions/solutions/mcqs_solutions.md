# Solutions to MCQs (Lecture 9)

1. **b) Policy Gradients can naturally handle continuous action spaces.** (By outputting parameters of a continuous distribution, like mean and variance of a Gaussian).
2. **b) It determines the direction in weight space that increases the probability of action $A_t$.** (Gradient ascent step).
3. **a) To reduce the variance of the gradient estimates without introducing bias.** (A state-dependent baseline shifts the Advantage, stabilizing the updates).
4. **b) To estimate the Value function $V(s)$ and compute the TD Error (Advantage) for the Actor.** (Bootstrapping to replace the noisy Monte Carlo return).
