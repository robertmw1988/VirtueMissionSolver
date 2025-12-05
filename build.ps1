<# 
.SYNOPSIS
    Build script for Virtue Mission Solver standalone executable.

.DESCRIPTION
    This script builds a standalone Windows executable using PyInstaller.
    The output is placed in the dist/ directory.

    Prerequisites:
    - Python 3.11+ with virtual environment set up (run setup.ps1 first)
    - PyInstaller installed (included in dev dependencies)

.PARAMETER Clean
    Remove build artifacts before building.

.EXAMPLE
    .\build.ps1
    # Build the executable

.EXAMPLE
    .\build.ps1 -Clean
    # Clean build artifacts and rebuild
#>

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "=== Virtue Mission Solver Build ===" -ForegroundColor Cyan
Write-Host ""

# Check for virtual environment
$venvPath = ".\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Error: Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first to create the virtual environment." -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

# Check PyInstaller is installed
$pyinstallerCheck = pip show pyinstaller 2>$null
if (-not $pyinstallerCheck) {
    Write-Host "PyInstaller not found. Installing..." -ForegroundColor Yellow
    pip install pyinstaller
}

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    Write-Host "Clean complete." -ForegroundColor Green
}

# Build the executable
Write-Host "Building standalone executable..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

pyinstaller virtue_mission_solver.spec --noconfirm

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Build Complete! ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output location:" -ForegroundColor Cyan
    Write-Host "    dist\VirtueMissionSolver.exe" -ForegroundColor White
    Write-Host ""
    
    # Show file size
    $exePath = "dist\VirtueMissionSolver.exe"
    if (Test-Path $exePath) {
        $size = (Get-Item $exePath).Length / 1MB
        Write-Host "File size: $([math]::Round($size, 2)) MB" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "You can distribute this single .exe file to users." -ForegroundColor Cyan
    Write-Host "No Python installation required on target machines." -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Build failed! Check the output above for errors." -ForegroundColor Red
    exit 1
}
