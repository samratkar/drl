"""P11. UCB Score Visualization — Heatmap"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    k, steps = 10, 200
    q_true = np.random.randn(k)
    Q = np.zeros(k)
    N = np.zeros(k)
    c = 2.0

    scores = np.zeros((k, steps))

    for t in range(steps):
        ucb_vals = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5))
        scores[:, t] = ucb_vals

        a = np.argmax(ucb_vals)
        r = np.random.randn() + q_true[a]
        N[a] += 1
        Q[a] += (1.0 / N[a]) * (r - Q[a])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(scores, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Arm", fontsize=12)
    ax.set_title("UCB Scores Over Time (c=2)", fontsize=13, fontweight='bold')
    ax.set_yticks(range(k))
    plt.colorbar(im, ax=ax, label="UCB Score")

    best = np.argmax(q_true)
    ax.annotate(f"← true best (arm {best})", xy=(steps - 1, best),
                fontsize=9, color='white', fontweight='bold',
                va='center', ha='right',
                bbox=dict(boxstyle='round', fc='black', alpha=0.6))

    plt.tight_layout()
    plt.savefig("p11_ucb_heatmap.png", dpi=150)
    plt.close()
    print("Saved: p11_ucb_heatmap.png")
