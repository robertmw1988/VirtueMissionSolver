# Virtue Mission Solver

A linear-programming mission optimizer and Bill of Materials (BOM) engine for Egg Inc. artifact missions.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Mission Optimizer** — LP-based solver to select optimal missions given fuel, time, and artifact priorities
- **BOM Engine** — Bill of Materials rollup for crafting artifacts from mission drops
- **GUI Application** — PySide6-based graphical interface for interactive optimization
- **CLI Tools** — Command-line interface for automation and scripting
- **Configurable** — User-configurable weights, constraints, and epic research settings
- **Standalone Executable** — Build a single .exe file for distribution (no Python required)

## Quick Start

### Option 1: Standalone Executable (Recommended for End Users)

Download `VirtueMissionSolver.exe` from the [Releases](https://github.com/robertmw1988/VirtueMissionSolver/releases) page and run it directly. No installation required.

### Option 2: Run from Source

#### Prerequisites

- Python 3.11 or higher
- Git (for cloning)

#### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/robertmw1988/VirtueMissionSolver.git
   cd VirtueMissionSolver
   ```

2. **Run the setup script:**

   **Windows (PowerShell):**
   ```powershell
   .\setup.ps1
   ```

   **Linux/macOS:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This creates a virtual environment, installs all dependencies, and prepares the project for use.

3. **Activate the virtual environment:**

   **Windows:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```

#### Usage

**Launch the GUI:**
```bash
virtue-solver-gui
# or
python -m Solver.gui_app
```

**Run the CLI solver:**
```bash
virtue-solver
# or
python -m Solver.run_solver --num-ships 3 --verbose
```

**BOM rollup from command line:**
```bash
python -m Solver.bom "Henerprise,Short,Gold Meteorite,20"
python -m Solver.bom "Henerprise,Epic,10" "Atreggies,Short,Book of Basan,5"
```

## Building Standalone Executable

To build a standalone Windows executable that can be distributed without requiring Python:

1. **Ensure setup is complete:**
   ```powershell
   .\setup.ps1
   ```

2. **Run the build script:**
   ```powershell
   .\build.ps1
   ```

3. **Find the output:**
   ```
   dist\VirtueMissionSolver.exe
   ```

The resulting `.exe` file includes all dependencies and can be run on any Windows machine without Python installed.

### Build Options

```powershell
# Clean build (removes previous build artifacts first)
.\build.ps1 -Clean
```

## Documentation

- [**Solver Module Documentation**](Solver/README.md) — Comprehensive API reference, examples, and testing guide
- [**FetchData Documentation**](FetchData/README.md) — Data fetching utilities for Egg Inc. game data
- [**Data Endpoints Reference**](FetchData/DataEndpoints.md) — API reference for game data endpoints

## Project Structure

```
VirtueMissionSolver/
├── Solver/              # Main application code
│   ├── gui/             # PySide6 GUI components
│   ├── tests/           # Pytest test suite
│   ├── resources.py     # Path utilities for packaging
│   ├── mission_solver.py
│   ├── bom.py
│   ├── config.py
│   └── ...
├── FetchData/           # Data fetching utilities
├── Wasmegg/             # Game data files (JSON)
├── scripts/             # Helper scripts
├── setup.ps1            # Windows setup script
├── setup.sh             # Unix setup script
├── build.ps1            # Windows build script
├── virtue_mission_solver.spec  # PyInstaller configuration
└── pyproject.toml       # Project configuration
```

## Configuration

The solver uses YAML configuration files. Copy and customize the default config:

```bash
cp Solver/DefaultUserConfig.yaml my_config.yaml
# Edit my_config.yaml with your preferences
python -m Solver.run_solver --config my_config.yaml
```

See [Solver/README.md](Solver/README.md) for detailed configuration options.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest Solver/tests/test_solver_scenarios.py -v
```

### Code Quality

```bash
# Lint with ruff
ruff check .

# Format with black
black .

# Type check with mypy
mypy Solver/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the works of others on the Egg, Inc. Discord https://discord.gg/egginc
  - @geroniberry 
  - @mennoo 
  - @carpetsage 
- Game data sourced from [wasmegg](https://wasmegg.netlify.app/) tools
  - https://github.com/carpetsage/egg (carpetsage)
-  Artifact data sourced from Menno
   - https://github.com/menno-egginc/eggincdatacollection-docs (menno)
   - Be kind - Do not hit his endpoint unless you ask
- Built with [PuLP](https://coin-or.github.io/pulp/) for linear programming
- GUI powered by [PySide6](https://doc.qt.io/qtforpython-6/)
