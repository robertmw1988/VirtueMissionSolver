#!/bin/bash
#
# Setup script for Virtue Mission Solver - creates virtual environment and installs dependencies.
#
# This script:
#   1. Creates a Python virtual environment (.venv)
#   2. Activates the virtual environment
#   3. Upgrades pip to the latest version
#   4. Installs the project in editable mode with all dependencies
#   5. Displays usage instructions
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "=== Virtue Mission Solver Setup ==="
echo ""

# Check Python version
echo "Checking Python installation..."

# Try python3 first, then python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.11 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Found: $PYTHON_VERSION"

# Extract version numbers
VERSION_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
VERSION_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$VERSION_MAJOR" -lt 3 ] || ([ "$VERSION_MAJOR" -eq 3 ] && [ "$VERSION_MINOR" -lt 11 ]); then
    echo "Error: Python 3.11 or higher is required. Found Python $VERSION_MAJOR.$VERSION_MINOR"
    exit 1
fi

# Create virtual environment
VENV_PATH=".venv"
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_PATH"
        echo "Creating new virtual environment..."
        $PYTHON_CMD -m venv "$VENV_PATH"
    fi
else
    echo "Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv "$VENV_PATH"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install project in editable mode
echo "Installing Virtue Mission Solver and dependencies..."
pip install -e ".[dev]"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "    source .venv/bin/activate"
echo ""
echo "Usage:"
echo "    virtue-solver-gui      # Launch the GUI application"
echo "    virtue-solver          # Run the CLI solver"
echo "    python -m Solver.bom  # BOM rollup tool"
echo ""
echo "For more information, see README.md"
