Param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Function Assert-DirectoryExists {
    Param([string]$Path)
    if (-not (Test-Path -Path $Path -PathType Container)) {
        Write-Error "Missing directory: $Path"
    } else {
        Write-Host "OK: $Path"
    }
}

Function Assert-FileExists {
    Param([string]$Path)
    if (-not (Test-Path -Path $Path -PathType Leaf)) {
        Write-Error "Missing file: $Path"
    } else {
        Write-Host "OK: $Path"
    }
}

Write-Host "Checking required directories..."
$dirs = @(
    'backend',
    'frontend',
    'configs',
    'datasets',
    'metrics',
    'templates',
    'scripts',
    'docs'
)

$files = @(
    'README.md',
    'LICENSE',
    '.gitignore'
)

$failed = $false
foreach ($d in $dirs) {
    try { Assert-DirectoryExists $d } catch { $failed = $true }
}

Write-Host "Checking required files..."
foreach ($f in $files) {
    try { Assert-FileExists $f } catch { $failed = $true }
}

if ($failed) {
    Write-Error "Project structure test: FAIL"
    exit 1
}

Write-Host "Project structure test: PASS"
exit 0
