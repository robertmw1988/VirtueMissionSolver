<# 
.SYNOPSIS
    Setup script for Virtue Mission Solver - creates virtual environment and installs dependencies.

.DESCRIPTION
    This script:
    1. Creates a Python virtual environment (.venv)
    2. Activates the virtual environment
    3. Upgrades pip to the latest version
    4. Installs the project in editable mode with all dependencies
    5. Displays usage instructions

.EXAMPLE
    .\setup.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "=== Virtue Mission Solver Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
    
    # Verify Python 3.11+
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
            Write-Host "Error: Python 3.11 or higher is required. Found Python $major.$minor" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "Error: Python not found. Please install Python 3.11 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment
$venvPath = ".\.venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists at $venvPath" -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
        Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
        python -m venv $venvPath
    }
} else {
    Write-Host "Creating virtual environment at $venvPath..." -ForegroundColor Yellow
    python -m venv $venvPath
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install project in editable mode
Write-Host "Installing Virtue Mission Solver and dependencies..." -ForegroundColor Yellow
pip install -e ".[dev]"

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in the future, run:" -ForegroundColor Cyan
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Usage:" -ForegroundColor Cyan
Write-Host "    virtue-solver-gui      # Launch the GUI application" -ForegroundColor White
Write-Host "    virtue-solver          # Run the CLI solver" -ForegroundColor White
Write-Host "    python -m Solver.bom  # BOM rollup tool" -ForegroundColor White
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Cyan
