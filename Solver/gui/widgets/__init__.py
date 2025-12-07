"""Widget subpackage for reusable GUI components."""

from .ship_config import ShipConfigWidget, StarRatingWidget
from .epic_research import EpicResearchWidget
from .constraints import ConstraintsWidget
from .results import (
    ResultsWidget,
    PlannerResultsWidget,
    ComparisonResultsWidget,
    FuelUsageWidget,
    format_fuel_amount,
)
from .artifact_weights import (
    MissionArtifactWeightsWidget,
    CraftedArtifactWeightsWidget,
)
from .artifact_categories import (
    MissionArtifactCategoryWidget,
    CraftedArtifactCategoryWidget,
    ArtifactCategoryWidget,
    ArtifactCategory,
)
from .artifact_table import CombinedArtifactTableWidget
from .mission_list import MissionListWidget
from .solution_history import SolutionHistoryWidget

# Alias for backward compatibility
CombinedArtifactWidget = CombinedArtifactTableWidget
from .cost_weights import CostWeightsWidget

__all__ = [
    "ShipConfigWidget",
    "StarRatingWidget", 
    "EpicResearchWidget",
    "ConstraintsWidget",
    "ResultsWidget",
    "PlannerResultsWidget",
    "ComparisonResultsWidget",
    "FuelUsageWidget",
    "format_fuel_amount",
    "MissionArtifactWeightsWidget",
    "CraftedArtifactWeightsWidget",
    "MissionArtifactCategoryWidget",
    "CraftedArtifactCategoryWidget",
    "ArtifactCategoryWidget",
    "ArtifactCategory",
    "CombinedArtifactTableWidget",
    "CombinedArtifactWidget",
    "CostWeightsWidget",
    "MissionListWidget",
    "SolutionHistoryWidget",
]
