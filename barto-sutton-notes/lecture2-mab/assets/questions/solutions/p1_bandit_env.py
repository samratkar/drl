"""P1. Implement a Simple Bandit Environment"""
import numpy as np

class BanditEnv:
    def __init__(self, k=10):
        self.k = k
        self.q_star = np.random.randn(k)

    def pull(self, a):
        return np.random.randn() + self.q_star[a]

    @property
    def optimal_action(self):
        return np.argmax(self.q_star)


if __name__ == "__main__":
    env = BanditEnv(k=10)
    print("True q_star:", np.round(env.q_star, 3))
    print("Optimal arm:", env.optimal_action)

    n_pulls = 1000
    sample_means = np.zeros(env.k)
    for a in range(env.k):
        rewards = [env.pull(a) for _ in range(n_pulls)]
        sample_means[a] = np.mean(rewards)

    print("\nVerification (1000 pulls per arm):")
    print(f"{'Arm':<5} {'q_star':>8} {'Sample mean':>12} {'Error':>8}")
    for a in range(env.k):
        err = abs(sample_means[a] - env.q_star[a])
        print(f"{a:<5} {env.q_star[a]:>8.3f} {sample_means[a]:>12.3f} {err:>8.3f}")
