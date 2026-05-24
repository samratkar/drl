"""P2. Sample-Average Action-Value Estimation"""
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


def run_greedy_sample_average(env, steps=1000):
    Q = np.zeros(env.k)
    N = np.zeros(env.k)

    for t in range(steps):
        a = np.argmax(Q) if t > 0 else np.random.randint(env.k)
        r = env.pull(a)
        N[a] += 1
        Q[a] += (1.0 / N[a]) * (r - Q[a])

    return Q, N


if __name__ == "__main__":
    env = BanditEnv(k=10)
    Q, N = run_greedy_sample_average(env, steps=1000)

    print(f"{'Arm':<5} {'q_star':>8} {'Q_est':>8} {'N_pulls':>8}")
    for a in range(env.k):
        print(f"{a:<5} {env.q_star[a]:>8.3f} {Q[a]:>8.3f} {int(N[a]):>8}")
    print(f"\nGreedy locked onto arm {np.argmax(N)} "
          f"(true best: arm {env.optimal_action})")
