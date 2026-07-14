# Solutions to MCQs (Chapter 9)

1. **b) The state space is continuous and too large.** (Tabular methods require a discrete table entry for every possible state).
2. **b) Updating the value of one state also updates the estimated values of similar states.** (Because states share weights in the function approximator).
3. **b) It ignores the effect of changing the weight vector on the target.** (The target $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$ depends on $\mathbf{w}$, but the gradient calculation acts as if the target is a fixed, independent constant).
4. **c) The feature vector $\mathbf{x}(s)$**. (Because $\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s)$).
5. **b) It is incredibly computationally efficient because exactly $m$ features are active at any time.** (We only need to sum $m$ weights, rather than computing exponential distances for every RBF feature).
