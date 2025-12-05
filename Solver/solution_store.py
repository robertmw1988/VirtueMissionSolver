"""
Solution persistence using YAML files.

Saves optimizer and mission planner solutions to user data directory
for recall and comparison.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Determine user data directory based on platform
def get_user_data_dir() -> Path:
    """
    Get the appropriate user data directory for the platform.
    
    Returns:
        Path to user data directory (created if needed)
    """
    app_name = "VirtueMissionSolver"
    
    if platform.system() == "Windows":
        # Use %APPDATA%/VirtueMissionSolver
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif platform.system() == "Darwin":
        # macOS: ~/Library/Application Support/VirtueMissionSolver
        base = Path.home() / "Library" / "Application Support"
    else:
        # Linux/Unix: ~/.local/share/VirtueMissionSolver
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    
    data_dir = base / app_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_solutions_dir() -> Path:
    """Get the solutions directory (created if needed)."""
    solutions_dir = get_user_data_dir() / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    return solutions_dir


# Source types for solutions
class SolutionSource:
    OPTIMIZER = "Solution"
    MISSION_LIST = "Mission-List"


@dataclass
class MissionListItem:
    """A single mission in a mission list."""
    ship: str  # API name (e.g., "HENERPRISE")
    ship_label: str  # Display name (e.g., "Henerprise")
    duration: str  # SHORT, LONG, EPIC
    level: int
    target: Optional[str]  # Target artifact or None for "Any"
    count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ship": self.ship,
            "ship_label": self.ship_label,
            "duration": self.duration,
            "level": self.level,
            "target": self.target,
            "count": self.count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionListItem":
        return cls(
            ship=data["ship"],
            ship_label=data.get("ship_label", data["ship"]),
            duration=data["duration"],
            level=data.get("level", 0),
            target=data.get("target"),
            count=data.get("count", 1),
        )


@dataclass
class SolutionSummary:
    """Summary of solver/calculation results."""
    status: str  # "Optimal", "Calculated", etc.
    objective_value: float
    total_time_hours: float
    total_drops: Dict[str, float] = field(default_factory=dict)
    crafted: Dict[str, float] = field(default_factory=dict)
    consumed: Dict[str, float] = field(default_factory=dict)
    remaining: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "objective_value": self.objective_value,
            "total_time_hours": self.total_time_hours,
            "total_drops": self.total_drops,
            "crafted": self.crafted,
            "consumed": self.consumed,
            "remaining": self.remaining,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolutionSummary":
        return cls(
            status=data.get("status", "Unknown"),
            objective_value=data.get("objective_value", 0.0),
            total_time_hours=data.get("total_time_hours", 0.0),
            total_drops=data.get("total_drops", {}),
            crafted=data.get("crafted", {}),
            consumed=data.get("consumed", {}),
            remaining=data.get("remaining", {}),
        )


@dataclass
class SavedSolution:
    """A saved solution with metadata."""
    name: str  # Auto-generated filename (without .yaml)
    display_name: str  # User-editable display name
    timestamp: str  # ISO format timestamp
    source_type: str  # "Solution" or "Mission-List"
    mission_list: List[MissionListItem]
    result: SolutionSummary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "timestamp": self.timestamp,
            "source_type": self.source_type,
            "mission_list": [m.to_dict() for m in self.mission_list],
            "result": self.result.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedSolution":
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            timestamp=data.get("timestamp", ""),
            source_type=data.get("source_type", SolutionSource.OPTIMIZER),
            mission_list=[MissionListItem.from_dict(m) for m in data.get("mission_list", [])],
            result=SolutionSummary.from_dict(data.get("result", {})),
        )


class SolutionStore:
    """
    Manages saving and loading solutions from YAML files.
    
    Solutions are stored in the user data directory under 'solutions/'.
    Each solution is a separate YAML file named {auto_name}.yaml.
    """
    
    def __init__(self, solutions_dir: Optional[Path] = None):
        """
        Initialize the solution store.
        
        Parameters
        ----------
        solutions_dir : Path, optional
            Custom directory for solutions. Defaults to user data dir.
        """
        self._solutions_dir = solutions_dir or get_solutions_dir()
        self._solutions_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def solutions_dir(self) -> Path:
        """Get the solutions directory path."""
        return self._solutions_dir
    
    def generate_name(self, source_type: str) -> str:
        """
        Generate an auto-name for a solution.
        
        Format: {source_type}_{YYYY-MM-DD}_{HHMMSS}
        Example: Solution_2024-12-04_143215
        """
        now = datetime.now()
        return f"{source_type}_{now.strftime('%Y-%m-%d_%H%M%S')}"
    
    def _get_solution_path(self, name: str) -> Path:
        """Get the file path for a solution by name."""
        return self._solutions_dir / f"{name}.yaml"
    
    def save_solution(self, solution: SavedSolution) -> Path:
        """
        Save a solution to a YAML file.
        
        Parameters
        ----------
        solution : SavedSolution
            The solution to save
            
        Returns
        -------
        Path
            Path to the saved file
        """
        path = self._get_solution_path(solution.name)
        
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                solution.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        
        return path
    
    def load_solution(self, name: str) -> Optional[SavedSolution]:
        """
        Load a solution from a YAML file.
        
        Parameters
        ----------
        name : str
            Solution name (filename without .yaml)
            
        Returns
        -------
        SavedSolution or None
            The loaded solution, or None if not found
        """
        path = self._get_solution_path(name)
        
        if not path.exists():
            return None
        
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return SavedSolution.from_dict(data)
        except (yaml.YAMLError, KeyError, TypeError) as e:
            print(f"Error loading solution {name}: {e}")
            return None
    
    def list_solutions(self) -> List[SavedSolution]:
        """
        List all saved solutions.
        
        Returns
        -------
        List[SavedSolution]
            All solutions, sorted by timestamp descending (newest first)
        """
        solutions = []
        
        for path in self._solutions_dir.glob("*.yaml"):
            name = path.stem
            solution = self.load_solution(name)
            if solution:
                solutions.append(solution)
        
        # Sort by timestamp descending
        solutions.sort(key=lambda s: s.timestamp, reverse=True)
        return solutions
    
    def delete_solution(self, name: str) -> bool:
        """
        Delete a solution file.
        
        Parameters
        ----------
        name : str
            Solution name to delete
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        path = self._get_solution_path(name)
        
        if path.exists():
            path.unlink()
            return True
        return False
    
    def rename_solution(self, name: str, new_display_name: str) -> bool:
        """
        Update the display name of a solution.
        
        Parameters
        ----------
        name : str
            Solution name (filename)
        new_display_name : str
            New display name to set
            
        Returns
        -------
        bool
            True if renamed successfully
        """
        solution = self.load_solution(name)
        if solution is None:
            return False
        
        solution.display_name = new_display_name
        self.save_solution(solution)
        return True
    
    def get_solutions_by_names(self, names: List[str]) -> List[SavedSolution]:
        """
        Load multiple solutions by name.
        
        Parameters
        ----------
        names : List[str]
            List of solution names to load
            
        Returns
        -------
        List[SavedSolution]
            Loaded solutions (skips any that fail to load)
        """
        solutions = []
        for name in names:
            solution = self.load_solution(name)
            if solution:
                solutions.append(solution)
        return solutions


# Module-level singleton
_solution_store: Optional[SolutionStore] = None


def get_solution_store() -> SolutionStore:
    """Get the singleton SolutionStore instance."""
    global _solution_store
    if _solution_store is None:
        _solution_store = SolutionStore()
    return _solution_store
