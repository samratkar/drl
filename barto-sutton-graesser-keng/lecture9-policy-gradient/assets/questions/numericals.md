# Numerical Questions (Lecture 9: Policy Gradients)

**Question 1: The REINFORCE Update**
An agent uses a Policy Network with a Softmax output layer. In state $S$, the network outputs the following probabilities for actions A, B, and C:
* $\pi(A|S) = 0.2$
* $\pi(B|S) = 0.5$
* $\pi(C|S) = 0.3$

The agent samples action **A**. It plays out the rest of the episode and receives a total return $G_t = +10$.
Assuming a baseline $V(S) = +4$ and a learning rate $\alpha = 0.1$.

1. Calculate the Advantage.
2. Will the probability of taking Action A increase or decrease after the gradient update? Why intuitively?
