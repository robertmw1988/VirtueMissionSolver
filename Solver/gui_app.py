#!/usr/bin/env python
"""
GUI entry point for the Virtue Mission Solver.

Usage:
    python -m Solver.gui_app
    virtue-solver-gui  (if installed via pip)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from .gui import MainWindow
from .mission_data import preload_mission_data


def main(config_path: Optional[Path] = None) -> int:
    """
    Launch the GUI application.
    
    Parameters
    ----------
    config_path : Path, optional
        Path to user config YAML. Defaults to Solver/DefaultUserConfig.yaml
    
    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Start background loading of mission data early
    preload_mission_data()
    
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Virtue Mission Solver")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("VirtueMissionSolver")
    
    # Apply a clean style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow(config_path)
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
