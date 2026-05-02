"""P9. Initial Bias Decay Curve"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    n_max = 50
    Q1 = 5.0
    alphas = [0.1, 0.2, 0.4]
    n = np.arange(n_max + 1)

    plt.figure(figsize=(10, 6))

    # Sample average: weight = 1 at n=0, then 0 for all n >= 1
    sa_weight = np.zeros(n_max + 1)
    sa_weight[0] = 1.0
    plt.plot(n, sa_weight, 'o-', color='#1971c2', label='Sample average (α=1/n)',
             markersize=3, linewidth=2)

    colors = ['#e8590c', '#d6336c', '#ae3ec9']
    for alpha, color in zip(alphas, colors):
        weight = (1 - alpha) ** n
        plt.plot(n, weight, '-', color=color, label=f'Constant α={alpha}',
                 linewidth=2)

    plt.xlabel("Steps (n)", fontsize=12)
    plt.ylabel("Weight on Q₁", fontsize=12)
    plt.title("Initial Bias Decay: How Fast Does Q₁ Disappear?", fontsize=13,
              fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("p9_bias_decay.png", dpi=150)
    plt.close()
    print("Saved: p9_bias_decay.png")
