---
layout: post
---

# Numerical Solutions (Lecture 11: Combined, Advanced PG & Model-Based RL)

1. **UCT Value Calculation:**
   Given: parent visit count $N(s) = 100$, child visit count $N(s,a) = 20$, total action value $W(s,a) = 12.0$, and exploration parameter $c = 1.414$.
   * **Average Action Value $Q(s,a)$:**
     $$ Q(s,a) = \frac{W(s,a)}{N(s,a)} = \frac{12.0}{20} = 0.6 $$
   * **Exploration Bonus:**
     $$ \text{Bonus} = c \sqrt{\frac{\ln N(s)}{N(s,a)}} = 1.414 \sqrt{\frac{\ln 100}{20}} = 1.414 \sqrt{\frac{4.605}{20}} = 1.414 \sqrt{0.230} \approx 1.414 \times 0.480 \approx 0.679 $$
   * **Total UCT Score:**
     $$ UCT(s,a) = Q(s,a) + \text{Bonus} \approx 0.6 + 0.679 = 1.279 $$

2. **AlphaGo Leaf Evaluation:**
   Given: Value Network estimate $v_{\theta}(s_L) = 0.75$, rollout outcome $z = -1.0$, and mixing parameter $\lambda = 0.5$.
   * **Combined Leaf Evaluation $V(s_L)$:**
     $$ V(s_L) = (1 - \lambda) v_{\theta}(s_L) + \lambda z = (1 - 0.5)(0.75) + 0.5(-1.0) = 0.375 - 0.500 = -0.125 $$
     The value backpropagated up the tree is $-0.125$.

3. **AlphaZero Loss Calculation:**
   Given: $z = 1.0$, $v_t = 0.6$, $\boldsymbol{\pi}_t = [0.1, 0.7, 0.2]^T$, $\mathbf{p}_t = [0.2, 0.5, 0.3]^T$.
   * **Value Squared Error Loss:**
     $$ \text{Value Loss} = (z - v_t)^2 = (1.0 - 0.6)^2 = (0.4)^2 = 0.16 $$
   * **Policy Cross-Entropy Loss:**
     $$ \text{Policy Loss} = -\sum_{a} \pi_a \ln p_a = - \left[ 0.1 \ln(0.2) + 0.7 \ln(0.5) + 0.2 \ln(0.3) \right] $$
     Using natural logs ($\ln(0.2) \approx -1.609$, $\ln(0.5) \approx -0.693$, $\ln(0.3) \approx -1.204$):
     $$ \text{Policy Loss} \approx - \left[ 0.1(-1.609) + 0.7(-0.693) + 0.2(-1.204) \right] $$
     $$ \text{Policy Loss} \approx - \left[ -0.161 - 0.485 - 0.241 \right] = -[-0.887] = 0.887 $$
   * **Total Loss (excluding L2 penalty):**
     $$ \text{Total Loss} = \text{Value Loss} + \text{Policy Loss} = 0.16 + 0.887 = 1.047 $$

4. **GAIL Surrogate Reward Calculation:**
   * **For expert-like transition ($D_{\phi}(s,a) = 0.90$):**
     $$ R(s,a) = -\ln(1 - 0.90) = -\ln(0.10) \approx -(-2.302) = 2.302 $$
   * **For agent-like transition ($D_{\phi}(s,a) = 0.10$):**
     $$ R(s,a) = -\ln(1 - 0.10) = -\ln(0.90) \approx -(-0.105) = 0.105 $$
     *Interpretation:* The agent receives a much higher reward ($2.302$) when it successfully fools the discriminator, and a very low reward ($0.105$) when the discriminator easily detects it as the generator.

5. **Behavior Cloning Gradient Update & Covariate Shift Analysis:**
   Given $x = 0.4, a^* = 0, w_0 = -0.5, \alpha = 2.0$:
   * **Policy Probabilities & Cross-Entropy Loss:**
     $$ w_0 \cdot x = (-0.5)(0.4) = -0.20 $$
     $$ P(a=1 \mid 0.4) = \sigma(-0.20) = \frac{1}{1 + e^{0.20}} \approx 0.4502 $$
     $$ P(a=0 \mid 0.4) = 1 - 0.4502 = 0.5498 $$
     $$ \mathcal{L}_{CE} = -\ln(0.5498) \approx 0.5982 $$
   * **Loss Gradient $\frac{\partial \mathcal{L}}{\partial w}$:**
     $$ \frac{\partial \mathcal{L}}{\partial w} = (P(a=1 \mid 0.4) - 0) \cdot 0.4 = (0.4502)(0.4) = +0.1801 $$
   * **Gradient Update ($w_1$):**
     $$ w_1 = w_0 - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w} = -0.5 - 2.0(0.1801) = -0.5 - 0.3602 = -0.8602 $$
   * **Gradient Magnitude at $x = +4.0$ vs. $x = +0.4$ (at $w = 0$):**
     * At $x = 0.4$: $\frac{\partial \mathcal{L}}{\partial w} = (0.5 - 0)(0.4) = +0.20$.
     * At $x = 4.0$: $\frac{\partial \mathcal{L}}{\partial w} = (0.5 - 0)(4.0) = +2.00$.
     * *Ratio:* The gradient magnitude at $x = +4.0$ is $\frac{2.00}{0.20} = 10\times$ larger.
     * *DAgger Mechanism:* Behavior Cloning fails because offline training data contains only near-center states ($x \approx 0.4$), so $x = +4.0$ is never trained on. When test-time perturbations push the agent to $x = +4.0$, BC makes errors that compound. DAgger explicitly rollouts the agent policy to visit $x = +4.0$, queries the expert for optimal action $a^*=0$, and injects this $10\times$ stronger gradient update into the dataset, rapidly learning robust recovery parameters ($w \ll 0$).

6. **Decision Transformer Return-Conditioned Action Prediction:**
   Given $\hat{R}_1 = 8.0, s_1 = 3.0$:
   * **Input Token Embeddings:**
     $$ \mathbf{e}_{R_1} = \mathbf{W}_R \cdot 8.0 = \begin{bmatrix} 0.5(8.0) \\ 0.0(8.0) \end{bmatrix} = \begin{bmatrix} 4.0 \\ 0.0 \end{bmatrix} $$
     $$ \mathbf{e}_{s_1} = \mathbf{W}_s \cdot 3.0 = \begin{bmatrix} 0.0(3.0) \\ 1.0(3.0) \end{bmatrix} = \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix} $$
   * **Raw Attention Scores for Token 2 ($s_1$):**
     $$ \mathbf{Q}_2 = \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix}, \quad \mathbf{K}_1 = \begin{bmatrix} 4.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{K}_2 = \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix} $$
     $$ S_{2,1} = \frac{\mathbf{Q}_2^T \mathbf{K}_1}{\sqrt{2}} = \frac{0.0(4.0) + 3.0(0.0)}{1.414} = 0.0 $$
     $$ S_{2,2} = \frac{\mathbf{Q}_2^T \mathbf{K}_2}{\sqrt{2}} = \frac{0.0(0.0) + 3.0(3.0)}{1.414} = \frac{9.0}{1.414} \approx 6.365 $$
   * **Causal Softmax Attention Weights:**
     $$ A_{2,1} = \frac{e^0}{e^0 + e^{6.365}} = \frac{1.0}{1.0 + 581.15} = \frac{1.0}{582.15} \approx 0.0017 $$
     $$ A_{2,2} = \frac{581.15}{582.15} \approx 0.9983 $$
   * **Contextual Representation $\mathbf{Z}_2$, Action Logits $\mathbf{z}$, and Probability:**
     $$ \mathbf{Z}_2 = 0.0017 \begin{bmatrix} 4.0 \\ 0.0 \end{bmatrix} + 0.9983 \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix} = \begin{bmatrix} 0.0068 \\ 2.9949 \end{bmatrix} $$
     $$ \mathbf{z} = \mathbf{W}_{\text{act}} \mathbf{Z}_2 = \begin{bmatrix} 1.0(0.0068) - 1.0(2.9949) \\ -1.0(0.0068) + 1.0(2.9949) \end{bmatrix} = \begin{bmatrix} -2.9881 \\ +2.9881 \end{bmatrix} $$
     $$ P(a=1 \mid s_1, \hat{R}_1) = \sigma(2.9881 - (-2.9881)) = \sigma(5.9762) \approx 0.9975 $$
     *Interpretation:* The target return prompt $\hat{R}_1 = 8.0$ conditions the causal self-attention layer to predict action $a=1$ with $99.75\%$ probability.
