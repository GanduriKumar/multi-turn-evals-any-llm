$ErrorActionPreference = 'Stop'

# Determine repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path $scriptDir

Write-Host "Starting backend and frontend as detached processes..."
Write-Host "These will continue running even if you close VS Code or lock your screen."
Write-Host ""

# Backend: start in a new persistent window
$backendCmd = "cd '$repoRoot'; `$env:PYTHONPATH='$repoRoot'; python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload --log-level debug"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Normal
Write-Host "Backend started in separate window on http://localhost:8000"

# Frontend: start in a new persistent window
$frontendCmd = "cd '$repoRoot\frontend'; $Env:DEBUG='vite:*'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -WindowStyle Normal
Write-Host "Frontend started in separate window on http://localhost:5173"

Write-Host ""
Write-Host "To stop: Close the PowerShell windows or use Stop-Process on the process IDs."
Write-Host "To find processes: Get-Process | Where-Object {`$_.ProcessName -like '*python*' -or `$_.ProcessName -like '*node*'}"
