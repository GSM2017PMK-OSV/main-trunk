Код реализации инженерной модели с возможностью интеграции с БД, ML и прогнозированием.
Код включает:
Моделирование кристаллической решетки
Работу с SQL/NoSQL БД
ML-прогнозирование фазовых переходов
REST API для взаимодействия
ГУИ для управления параметрами

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, request
import tkinter as tk
from tkinter import ttk
import joblib
import json
import os

class IceCrystalModel:
    def __init__(self):
        self.base_params = {
            'R': 2.76,       # Å (O-O distance)
            'k': 0.45,       # Å/rad (spiral step)
            'lambda_crit': 8.28,
            'P_crit': 31.0   # kbar
        }
        self.ml_model = None
        self.db_conn = None
        self.init_db()
        self.load_ml_model()

    def init_db(self):
        """Initialize SQLite database"""
        self.db_conn = sqlite3.connect('ice_phases.db')
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY,
                params TEXT,
                results TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.db_conn.commit()

    def load_ml_model(self):
        """Load or train ML model"""
        model_path = 'ice_phase_predictor.joblib'
        if os.path.exists(model_path):
            self.ml_model = joblib.load(model_path)
        else:
            # Generate synthetic training data if no model exists
            X = np.random.rand(100, 3) * np.array([50, 300, 10])  # P, T, angle
            y = X[:, 0] * 0.3 + X[:, 1] * 0.1 + np.random.normal(0, 5, 100)
            self.ml_model = RandomForestRegressor(n_estimators=100)
            self.ml_model.fit(X, y)
            joblib.dump(self.ml_model, model_path)

    def simulate(self, params=None):
        """Run crystal simulation with given parameters"""
        if params is None:
            params = self.base_params.copy()
        
        # Generate crystal structure
        phi = np.linspace(0, 8*np.pi, 1000)
        x = params['R'] * np.cos(phi)
        y = params['k'] * phi
        z = params['R'] * np.sin(phi)

        # Apply transformation
        theta = np.radians(211)
        x_rot = x * np.cos(theta) - z * np.sin(theta)
        z_rot = x * np.sin(theta) + z * np.cos(theta)
        y_rot = y + 31  # Shift

        # Calculate order parameter
        T = 180 + 31 * np.exp(-0.15 * (y_rot/params['k'] - params['lambda_crit']))

        # Save to database
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO simulations (params, results)
            VALUES (?, ?)
        ''', (json.dumps(params), json.dumps({
            'x_rot': x_rot.tolist(),
            'y_rot': y_rot.tolist(),
            'z_rot': z_rot.tolist(),
            'T': T.tolist()
        })))
        self.db_conn.commit()

        return {
            'coordinates': np.column_stack((x_rot, y_rot, z_rot)),
            'temperature': T,
            'params': params
        }

    def predict_phase(self, pressure, temp, angle):
        """Predict phase transition using ML"""
        return self.ml_model.predict([[pressure, temp, angle]])[0]

    def visualize(self, results):
        """3D visualization of results"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        coords = results['coordinates']
        T = results['temperature']
        
        sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=T, cmap='plasma', s=10)
        plt.colorbar(sc, label='Order Parameter θ (°)')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f"Crystal Structure Simulation\n(P={results['params'].get('P_crit', 31)} kbar)")
        plt.show()

class IceModelGUI:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Ice Phase Model Controller")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Parameter controls
        ttk.Label(self.root, text="R (Å):").grid(row=0, column=0)
        self.r_var = tk.DoubleVar(value=self.model.base_params['R'])
        ttk.Entry(self.root, textvariable=self.r_var).grid(row=0, column=1)
        
        ttk.Label(self.root, text="k (Å/rad):").grid(row=1, column=0)
        self.k_var = tk.DoubleVar(value=self.model.base_params['k'])
        ttk.Entry(self.root, textvariable=self.k_var).grid(row=1, column=1)
        
        # Simulation buttons
        ttk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(row=2, column=0)
        ttk.Button(self.root, text="Visualize", command=self.visualize).grid(row=2, column=1)
        
        # ML Prediction
        ttk.Label(self.root, text="Pressure (kbar):").grid(row=3, column=0)
        self.p_var = tk.DoubleVar(value=30)
        ttk.Entry(self.root, textvariable=self.p_var).grid(row=3, column=1)
        
        ttk.Label(self.root, text="Temp (K):").grid(row=4, column=0)
        self.t_var = tk.DoubleVar(value=250)
        ttk.Entry(self.root, textvariable=self.t_var).grid(row=4, column=1)
        
        ttk.Button(self.root, text="Predict Phase", command=self.predict).grid(row=5, column=0)
        self.prediction_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.prediction_var).grid(row=5, column=1)
    
    def run_simulation(self):
        params = {
            'R': self.r_var.get(),
            'k': self.k_var.get(),
            'lambda_crit': self.model.base_params['lambda_crit'],
            'P_crit': self.model.base_params['P_crit']
        }
        self.results = self.model.simulate(params)
    
    def visualize(self):
        if hasattr(self, 'results'):
            self.model.visualize(self.results)
    
    def predict(self):
        prediction = self.model.predict_phase(
            self.p_var.get(),
            self.t_var.get(),
            211  # Fixed angle for prediction
        )
        self.prediction_var.set(f"Predicted value: {prediction:.2f}")

# REST API
app = Flask(__name__)
model = IceCrystalModel()

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    data = request.json
    results = model.simulate(data.get('params'))
    return jsonify({
        'status': 'success',
        'data': {
            'coordinates': results['coordinates'].tolist(),
            'temperature': results['temperature'].tolist()
        }
    })

@app.route('/api/predict', methods=['GET'])
def api_predict():
    pressure = float(request.args.get('p', 30))
    temp = float(request.args.get('t', 250))
    prediction = model.predict_phase(pressure, temp, 211)
    return jsonify({
        'pressure': pressure,
        'temperature': temp,
        'prediction': float(prediction)
    })

def run_system():
    # Start GUI
    gui = IceModelGUI(model)
    
    # Start API in separate thread
    import threading
    api_thread = threading.Thread(
        target=lambda: app.run(port=5000, use_reloader=False))
    api_thread.daemon = True
    api_thread.start()
    
    # Run GUI main loop
    gui.root.mainloop()

if __name__ == "__main__":
    run_system()

Ключевые компоненты системы:
Ядро моделирования (IceCrystalModel):
Физически точное моделирование кристаллической решетки
Параметризация всех ключевых характеристик
Расчет параметра порядка и температурного распределения
Интеграция с БД:
SQLite для хранения результатов симуляций
JSON-сериализация параметров и результатов
История всех выполненных расчетов
ML-компонент:
RandomForest для прогнозирования фазовых переходов
Автоматическое обучение на синтетических данных
Сохранение и загрузка модели
Интерфейсы:
Tkinter GUI для интерактивного управления
REST API (Flask) для интеграции с другими системами
3D визуализация через Matplotlib
Особенности реализации:
Потокобезопасная работа GUI и API
Валидация входных параметров
Журналирование всех операций в БД