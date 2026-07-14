"""P7. Weight Visualization for Different Alpha Values"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    n = 20
    alphas = [0.1, 0.3, 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, alpha in zip(axes, alphas):
        weights = [alpha * (1 - alpha)**(n - i) for i in range(1, n + 1)]
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, n))
        ax.bar(range(1, n + 1), weights, color=colors, edgecolor='white',
               linewidth=0.5)
        ax.set_xlabel("Reward index i")
        ax.set_title(f"α = {alpha}")
        ax.set_xlim(0, n + 1)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel("Weight on Rᵢ")
    fig.suptitle(
        f"Exponential Recency Weights: α(1-α)^(n-i) for n={n} rewards",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("p7_weight_visualization.png", dpi=150)
    plt.close()
    print("Saved: p7_weight_visualization.png")
