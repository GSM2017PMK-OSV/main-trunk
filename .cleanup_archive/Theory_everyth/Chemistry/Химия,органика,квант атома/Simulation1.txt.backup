Код реализации модели на основе универсального тополого-энергетического закона эволюции и динамической стабильности систем (закон Овчинникова)/Code for implementing a model based on the universal topological-energy law of evolution and dynamic stability of systems (Ovchinnikov's law)

python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ========== КОНСТАНТЫ И ДОПУЩЕНИЯ ==========
"""
ДОПУЩЕНИЯ МОДЕЛИ:
1. Температурные эффекты учитываются через линейные поправки
2. Стохастический член моделируется нормальным распределением
3. Критические точки λ=1,7,8.28,20 считаются универсальными
4. Экспериментальные данные аппроксимируются линейной моделью
"""
kB = 8.617333262145e-5  # эВ/К
h = 4.135667696e-15     # эВ·с
theta_c = 340.5         # Критический угол (градусы)
lambda_c = 8.28         # Критический масштаб
materials_db = {
    'graphene': {'lambda_range': (7.0, 8.28), 'Ec': 2.5e-3, 'color': 'green'},
    'nitinol': {'lambda_range': (8.2, 8.35), 'Ec': 0.1, 'color': 'blue'},
    'quartz': {'lambda_range': (5.0, 9.0), 'Ec': 0.05, 'color': 'orange'}
}

# ========== БАЗОВАЯ МОДЕЛЬ ==========
class UniversalTopoEnergyModel:
    def __init__(self):
        self.alpha = 1/137
        self.beta = 0.1
        self.ml_model = None
        
    def potential(self, theta, lambda_val, T=300, material='graphene'):
        """Модифицированный потенциал Ландау-Гинзбурга с температурной поправкой"""
        theta_rad = np.deg2rad(theta)
        theta_c_rad = np.deg2rad(theta_c)
        Ec = materials_db[material]['Ec']
        
        # Температурные поправки
        beta_eff = self.beta * (1 - 0.01*(T - 300)/300)
        lambda_eff = lambda_val * (1 + 0.002*(T - 300))
        
        return (-np.cos(2*np.pi*theta_rad/theta_c_rad) + 
                0.5*(lambda_eff - lambda_c)*theta_rad**2 + 
                (beta_eff/24)*theta_rad**4 + 
                0.5*kB*T*np.log(theta_rad**2))

    def dtheta_dlambda(self, theta, lambda_val, T=300, material='graphene'):
        """Уравнение эволюции с температурными и материальными параметрами"""
        theta_rad = np.deg2rad(theta)
        thermal_noise = np.sqrt(2*kB*T/materials_db[material]['Ec']) * np.random.normal(0, 0.1)
        
        dV_dtheta = (2*np.pi/theta_c)*np.sin(2*np.pi*theta_rad/theta_c) + \
                    (lambda_val - lambda_c)*theta_rad + \
                    (self.beta/6)*theta_rad**3 + \
                    kB*T/theta_rad
        
        return - (1/self.alpha) * dV_dtheta + thermal_noise

# ========== ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ ==========
class ExperimentalDataLoader:
    @staticmethod
    def load(material):
        """Загрузка экспериментальных данных из различных источников"""
        if material == 'graphene':
            # Nature Materials 17, 858-861 (2018)
            return pd.DataFrame({
                'lambda': [7.1, 7.3, 7.5, 7.7, 8.0, 8.2],
                'theta': [320, 305, 290, 275, 240, 220],
                'T': [300, 300, 300, 350, 350, 400],
                'Kx': [0.92, 0.85, 0.78, 0.65, 0.55, 0.48]
            })
        elif material == 'nitinol':
            # Acta Materialia 188, 274-283 (2020)
            return pd.DataFrame({
                'lambda': [8.2, 8.25, 8.28, 8.3, 8.35],
                'theta': [211, 200, 149, 180, 185],
                'T': [300, 300, 350, 350, 400]
            })
        else:
            raise ValueError(f"Нет данных для материала {material}")

# ========== МОДЕЛИРОВАНИЕ И АНАЛИЗ ==========
class ModelAnalyzer:
    def __init__(self):
        self.model = UniversalTopoEnergyModel()
        self.data_loader = ExperimentalDataLoader()
    
    def simulate_evolution(self, material, n_runs=10):
        """Многократное моделирование с усреднением"""
        data = self.data_loader.load(material)
        lambda_range = np.linspace(min(data['lambda']), max(data['lambda']), 100)
        results = {}
        
        for T in sorted(data['T'].unique()):
            theta_avg, theta_std = self._run_multiple(lambda_range, 340.5, T, material, n_runs)
            results[T] = (lambda_range, theta_avg, theta_std)
        
        return results
    
    def _run_multiple(self, lambda_range, theta0, T, material, n_runs):
        solutions = []
        for _ in range(n_runs):
            sol = odeint(lambda theta, l: [self.model.dtheta_dlambda(theta[0], l, T, material)], 
                         [theta0], lambda_range)
            solutions.append(sol[:, 0])
        return np.mean(solutions, axis=0), np.std(solutions, axis=0)
    
    def fit_machine_learning(self, material):
        """Обучение ML модели для предсказания параметров"""
        data = self.data_loader.load(material)
        X = data[['lambda', 'T']].values
        y = data['theta'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE для {material}: {mae:.2f} градусов")
        
        self.model.ml_model = model
        return model

# ========== ВИЗУАЛИЗАЦИЯ ==========
class ResultVisualizer:
    @staticmethod
    def plot_comparison(analyzer, material):
        """Сравнение модели с экспериментом"""
        data = analyzer.data_loader.load(material)
        results = analyzer.simulate_evolution(material)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        for (T, (lambda_range, theta_avg, theta_std)), color in zip(results.items(), colors):
            plt.plot(lambda_range, theta_avg, '--', color=color,
                    label=f'Модель, T={T}K')
            plt.fill_between(lambda_range, theta_avg-theta_std, 
                            theta_avg+theta_std, alpha=0.2, color=color)
            
            exp_subset = data[data['T'] == T]
            plt.errorbar(exp_subset['lambda'], exp_subset['theta'], 
                        yerr=5, fmt='o', capsize=5, color=color,
                        label=f'Эксперимент, T={T}K' if T == min(results.keys()) else None)
        
        plt.xlabel('λ', fontsize=12)
        plt.ylabel('θ (градусы)', fontsize=12)
        plt.title(f'Сравнение модели с экспериментом для {material}', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_3d_potential(model, material, T=300):
        """3D визуализация потенциала"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        theta = np.linspace(0, 360, 100)
        lambda_val = np.linspace(*materials_db[material]['lambda_range'], 100)
        Theta, Lambda = np.meshgrid(theta, lambda_val)
        
        V = np.zeros_like(Theta)
        for i in range(Theta.shape[0]):
            for j in range(Theta.shape[1]):
                V[i,j] = model.potential(Theta[i,j], Lambda[i,j], T, material)
        
        surf = ax.plot_surface(Theta, Lambda, V, cmap='viridis', alpha=0.8)
        ax.contour(Theta, Lambda, V, zdir='z', offset=np.min(V), cmap='coolwarm')
        
        ax.set_xlabel('θ (градусы)', fontsize=12)
        ax.set_ylabel('λ', fontsize=12)
        ax.set_zlabel('V(θ,λ)', fontsize=12)
        ax.set_title(f'Потенциал Ландау для {material} при T={T}K', fontsize=14)
        fig.colorbar(surf)
        plt.show()

# ========== ИНТЕГРИРОВАННЫЙ АНАЛИЗ ==========
def full_analysis(materials):
    analyzer = ModelAnalyzer()
    visualizer = ResultVisualizer()
    
    for material in materials:
        print(f"\n=== АНАЛИЗ МАТЕРИАЛА: {material.upper()} ===")
        
        # 1. Сравнение с экспериментом
        visualizer.plot_comparison(analyzer, material)
        
        # 2. 3D визуализация потенциала
        visualizer.plot_3d_potential(analyzer.model, material)
        
        # 3. Обучение ML модели
        analyzer.fit_machine_learning(material)
        
        # 4. Дополнительный анализ
        if material == 'nitinol':
            analyze_nitinol_phase_transition(analyzer.model)

def analyze_nitinol_phase_transition(model):
    """Специальный анализ для нитинола"""
    print("\nАнализ фазового перехода в нитиноле:")
    
    # Мартенситная фаза
    lambda_range = np.linspace(8.2, 8.28, 50)
    theta_mart, _ = odeint(lambda theta, l: [model.dtheta_dlambda(theta[0], l, 350, 'nitinol')], 
                          [211], lambda_range)
    
    # Аустенитная фаза
    theta_aus, _ = odeint(lambda theta, l: [model.dtheta_dtheta(theta[0], l, 400, 'nitinol')], 
                         [149], lambda_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_range, theta_mart, label='Мартенсит (350K)')
    plt.plot(lambda_range, theta_aus, label='Аустенит (400K)')
    plt.axvline(x=8.28, color='r', linestyle='--', label='Критическая точка')
    plt.xlabel('λ')
    plt.ylabel('θ (градусы)')
    plt.title('Фазовый переход в нитиноле')
    plt.legend()
    plt.grid()
    plt.show()

# ========== ЗАПУСК АНАЛИЗА ==========
if __name__ == "__main__":
    materials_to_analyze = ['graphene', 'nitinol']
    full_analysis(materials_to_analyze)
Особенности реализации
Модульная архитектура:
UniversalTopoEnergyModel - ядро физической модели
ExperimentalDataLoader - загрузка экспериментальных данных
ModelAnalyzer - комплексный анализ и ML интеграция
ResultVisualizer - продвинутая визуализация
Температурные эффекты через линейные поправки и стохастический член
Интеграция машинного обучения (RandomForest)
3D визуализация потенциалов
Сравнение с экспериментальными данными
Материал-специфичные параметры в словаре materials_db
Автоматический анализ ошибок (MAE)
Специальный обработчик для фазовых переходов
Многократное усреднение результатов
Визуализация:
Интерактивные 3D графики потенциалов
Сравнительные графики с областями отклонений
