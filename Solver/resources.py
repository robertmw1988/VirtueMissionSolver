"""
Resource path utilities for frozen (PyInstaller) and development modes.

This module provides a single function to resolve paths to bundled resources,
working correctly both during development and when packaged as a standalone
executable with PyInstaller.
"""
from __future__ import annotations

import sys
from pathlib import Path


def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and PyInstaller.
    
    In development mode, paths are resolved relative to the project root.
    In frozen (PyInstaller) mode, paths are resolved relative to the
    temporary extraction directory (_MEIPASS).
    
    Parameters
    ----------
    relative_path : str
        Path relative to project root (e.g., "Wasmegg/eiafx-data.json")
    
    Returns
    -------
    Path
        Absolute path to the resource
    
    Examples
    --------
    >>> config_path = get_resource_path("Solver/DefaultUserConfig.yaml")
    >>> data_path = get_resource_path("Wasmegg/eiafx-data.json")
    """
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        # sys._MEIPASS is the path to the temp folder where PyInstaller extracts files
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # Running in development - this file is in Solver/, so parent.parent is project root
        base_path = Path(__file__).resolve().parent.parent
    return base_path / relative_path


def is_frozen() -> bool:
    """
    Check if running in a frozen (PyInstaller) environment.
    
    Returns
    -------
    bool
        True if running as a packaged executable, False in development
    """
    return getattr(sys, 'frozen', False)
