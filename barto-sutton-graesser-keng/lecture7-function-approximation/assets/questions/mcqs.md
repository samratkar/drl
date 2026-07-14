---
layout: post
---

# Multiple Choice Questions (Chapter 9: Function Approximation)

1. Why do tabular methods fail in environments like autonomous driving?
   a) They cannot handle sparse rewards.
   b) The state space is continuous and too large.
   c) They require too much exploration.
   d) They are off-policy.

2. In the context of function approximation, what does the term "generalization" mean?
   a) The agent learns to ignore negative rewards.
   b) Updating the value of one state also updates the estimated values of similar states.
   c) The agent uses a general policy instead of an optimal one.
   d) The algorithm works across multiple different environments without retraining.

3. Why is standard TD(0) with function approximation referred to as a "semi-gradient" method?
   a) It only takes half a step towards the gradient.
   b) It ignores the effect of changing the weight vector on the target.
   c) It uses a stochastic estimate of the gradient rather than the true gradient.
   d) It only approximates the value function, not the policy.

4. For linear function approximation, the gradient $\nabla \hat{v}(s, \mathbf{w})$ is exactly equal to:
   a) The weight vector $\mathbf{w}$
   b) The step size $\alpha$
   c) The feature vector $\mathbf{x}(s)$
   d) The TD error $\delta$

5. Which of the following is an advantage of Tile Coding over radial basis functions (RBFs)?
   a) It provides a globally smooth, infinitely differentiable value function.
   b) It is incredibly computationally efficient because exactly $m$ features are active at any time.
   c) It eliminates the need for a step-size parameter $\alpha$.
   d) It automatically discovers the underlying non-linear features of the environment.
