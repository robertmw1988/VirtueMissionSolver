"""
Results display widgets.

Displays solver output including:
- Mission recommendations table
- Expected artifact drops
- BOM rollup (crafting summary)
- Fuel usage summary by egg type
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QScrollArea,
    QFrame,
    QTextEdit,
    QSplitter,
    QTabWidget,
    QGridLayout,
)

from ...mission_solver import SolverResult, FuelUsage
from ...bom import RollupResult
from ...aliases import get_egg_display_name


def format_fuel_amount(amount: float) -> str:
    """Format fuel amount with appropriate suffix (T/B/M)."""
    if amount >= 1e12:
        return f"{amount / 1e12:.2f}T"
    elif amount >= 1e9:
        return f"{amount / 1e9:.2f}B"
    elif amount >= 1e6:
        return f"{amount / 1e6:.2f}M"
    elif amount >= 1e3:
        return f"{amount / 1e3:.2f}K"
    else:
        return f"{amount:,.0f}"


class FuelUsageWidget(QWidget):
    """
    Widget displaying fuel usage breakdown by egg type.
    
    Shows each virtue egg used with amounts and a total tank usage.
    """
    
    # Egg display order and colors
    EGG_COLORS = {
        "HUMILITY": "#B1B1B1",     # Brown
        "INTEGRITY": "#A3844E",    # Royal Blue
        "CURIOSITY": "#2A2A2A",    # Dark Orchid
        "KINDNESS": "#3A5EEF",     # Hot Pink
        "RESILIENCE": "#EAD84A",   # Forest Green
    }
    
    EGG_ORDER = ["HUMILITY", "INTEGRITY", "CURIOSITY", "KINDNESS", "RESILIENCE"]
    
    def __init__(self, compact: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._compact = compact
        self._fuel_labels: Dict[str, QLabel] = {}
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        if not self._compact:
            # Header
            header = QLabel("Fuel Usage by Egg")
            header.setStyleSheet("font-weight: bold; font-size: 12px;")
            layout.addWidget(header)
        
        # Grid for egg amounts - tight layout
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)
        # Don't let columns stretch
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        
        row = 0
        for egg in self.EGG_ORDER:
            color = self.EGG_COLORS.get(egg, "#666")
            display_name = get_egg_display_name(egg).replace(" Egg", "")
            
            # Egg name label
            name_label = QLabel(f"{display_name}:")
            name_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            # Amount label - fixed width for alignment
            amount_label = QLabel("-")
            amount_label.setStyleSheet("font-family: monospace;")
            amount_label.setMinimumWidth(60)
            self._fuel_labels[egg] = amount_label
            
            grid.addWidget(name_label, row, 0, Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(amount_label, row, 1, Qt.AlignmentFlag.AlignLeft)
            row += 1
        
        # Wrap grid in a horizontal layout to prevent stretching
        grid_container = QHBoxLayout()
        grid_container.setContentsMargins(0, 0, 0, 0)
        grid_container.addLayout(grid)
        grid_container.addStretch(1)  # Push everything to the left
        
        layout.addLayout(grid_container)
        
        if not self._compact:
            # Tank total
            self._tank_total_label = QLabel("Tank Total: -")
            self._tank_total_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
            layout.addWidget(self._tank_total_label)
        
        # Add stretch at bottom to top-align everything
        layout.addStretch(1)
    
    def clear(self) -> None:
        """Reset all labels."""
        for label in self._fuel_labels.values():
            label.setText("-")
        if hasattr(self, "_tank_total_label"):
            self._tank_total_label.setText("Tank Total: -")
    
    def set_fuel_usage(self, fuel_usage: FuelUsage) -> None:
        """Update display from FuelUsage object."""
        for egg in self.EGG_ORDER:
            amount = fuel_usage.by_egg.get(egg, 0)
            if amount > 0:
                self._fuel_labels[egg].setText(format_fuel_amount(amount))
            else:
                self._fuel_labels[egg].setText("-")
        
        if hasattr(self, "_tank_total_label"):
            tank_total = fuel_usage.tank_total
            self._tank_total_label.setText(f"Tank Total: {format_fuel_amount(tank_total)}")
    
    def set_fuel_dict(self, fuel_by_egg: Dict[str, float]) -> None:
        """Update display from a dict of egg -> amount."""
        for egg in self.EGG_ORDER:
            amount = fuel_by_egg.get(egg, 0)
            if amount > 0:
                self._fuel_labels[egg].setText(format_fuel_amount(amount))
            else:
                self._fuel_labels[egg].setText("-")
        
        if hasattr(self, "_tank_total_label"):
            # Tank eggs exclude HUMILITY
            tank_eggs = {"INTEGRITY", "CURIOSITY", "KINDNESS", "RESILIENCE"}
            tank_total = sum(fuel_by_egg.get(e, 0) for e in tank_eggs)
            self._tank_total_label.setText(f"Tank Total: {format_fuel_amount(tank_total)}")


class MissionTableWidget(QWidget):
    """
    Table displaying recommended missions.
    
    Columns: Count, Ship, Duration, Level, Target
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("Recommended Missions")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Count", "Ship", "Duration", "Level", "Target"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)
    
    def clear(self) -> None:
        """Clear the table."""
        self._table.setRowCount(0)
    
    def set_results(self, result: SolverResult) -> None:
        """Populate table from solver result."""
        self.clear()
        
        missions = result.selected_missions
        self._table.setRowCount(len(missions))
        
        for row, (mission, count) in enumerate(missions):
            # Count
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 0, count_item)
            
            # Ship
            self._table.setItem(row, 1, QTableWidgetItem(mission.ship_label))
            
            # Duration
            duration = mission.duration_type.capitalize()
            self._table.setItem(row, 2, QTableWidgetItem(duration))
            
            # Level
            level_item = QTableWidgetItem(str(mission.level))
            level_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 3, level_item)
            
            # Target
            target = mission.target_artifact or "Any"
            if target.upper() == "UNKNOWN":
                target = "Any"
            self._table.setItem(row, 4, QTableWidgetItem(target))


class DropsTableWidget(QWidget):
    """
    Table displaying expected artifact drops.
    
    Columns: Artifact, Expected Count
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("Expected Drops")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Artifact", "Expected"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)
    
    def clear(self) -> None:
        """Clear the table."""
        self._table.setRowCount(0)
    
    def set_drops(self, drops: dict[str, float]) -> None:
        """Populate table from drops dict."""
        self.clear()
        
        # Sort by count descending, filter out zero
        sorted_drops = sorted(
            [(art, amt) for art, amt in drops.items() if amt > 0.01],
            key=lambda x: -x[1]
        )
        
        self._table.setRowCount(len(sorted_drops))
        
        for row, (artifact, amount) in enumerate(sorted_drops):
            self._table.setItem(row, 0, QTableWidgetItem(artifact))
            
            amt_item = QTableWidgetItem(f"{amount:.2f}")
            amt_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 1, amt_item)


class BOMRollupWidget(QWidget):
    """
    Table displaying BOM rollup results - what can be crafted from drops.
    
    Shows:
    - Crafted artifacts and quantities
    - Consumed ingredients
    - Remaining inventory
    - Any shortfalls
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Sub-tabs for different BOM sections
        self._bom_tabs = QTabWidget()
        
        # Crafted artifacts tab
        self._crafted_table = QTableWidget()
        self._crafted_table.setColumnCount(2)
        self._crafted_table.setHorizontalHeaderLabels(["Artifact", "Crafted"])
        self._crafted_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._crafted_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._crafted_table.setAlternatingRowColors(True)
        self._crafted_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._bom_tabs.addTab(self._crafted_table, "Crafted")
        
        # Consumed ingredients tab
        self._consumed_table = QTableWidget()
        self._consumed_table.setColumnCount(2)
        self._consumed_table.setHorizontalHeaderLabels(["Ingredient", "Used"])
        self._consumed_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._consumed_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._consumed_table.setAlternatingRowColors(True)
        self._consumed_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._bom_tabs.addTab(self._consumed_table, "Consumed")
        
        # Remaining inventory tab
        self._remaining_table = QTableWidget()
        self._remaining_table.setColumnCount(2)
        self._remaining_table.setHorizontalHeaderLabels(["Artifact", "Remaining"])
        self._remaining_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._remaining_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._remaining_table.setAlternatingRowColors(True)
        self._remaining_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._bom_tabs.addTab(self._remaining_table, "Remaining")
        
        layout.addWidget(self._bom_tabs)
        
        # Placeholder label when no BOM data
        self._placeholder = QLabel("No BOM rollup data.\nSet crafting weights to enable.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self._placeholder)
        
        # Initially show placeholder
        self._bom_tabs.hide()
    
    def clear(self) -> None:
        """Clear all tables."""
        self._crafted_table.setRowCount(0)
        self._consumed_table.setRowCount(0)
        self._remaining_table.setRowCount(0)
        self._bom_tabs.hide()
        self._placeholder.show()
    
    def set_rollup(self, rollup: Optional[RollupResult]) -> None:
        """Populate tables from BOM rollup result."""
        self.clear()
        
        if rollup is None:
            return
        
        # Check if there's any data
        has_data = (rollup.crafted or rollup.consumed or rollup.remaining)
        if not has_data:
            return
        
        self._placeholder.hide()
        self._bom_tabs.show()
        
        # Populate crafted table
        if rollup.crafted:
            sorted_crafted = sorted(
                [(art, qty) for art, qty in rollup.crafted.items() if qty > 0.001],
                key=lambda x: -x[1]
            )
            self._crafted_table.setRowCount(len(sorted_crafted))
            for row, (artifact, qty) in enumerate(sorted_crafted):
                self._crafted_table.setItem(row, 0, QTableWidgetItem(artifact))
                qty_item = QTableWidgetItem(f"{qty:.2f}")
                qty_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._crafted_table.setItem(row, 1, qty_item)
        
        # Populate consumed table
        if rollup.consumed:
            sorted_consumed = sorted(
                [(art, qty) for art, qty in rollup.consumed.items() if qty > 0.001],
                key=lambda x: -x[1]
            )
            self._consumed_table.setRowCount(len(sorted_consumed))
            for row, (artifact, qty) in enumerate(sorted_consumed):
                self._consumed_table.setItem(row, 0, QTableWidgetItem(artifact))
                qty_item = QTableWidgetItem(f"{qty:.2f}")
                qty_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._consumed_table.setItem(row, 1, qty_item)
        
        # Populate remaining table
        if rollup.remaining:
            sorted_remaining = sorted(
                [(art, qty) for art, qty in rollup.remaining.items() if qty > 0.001],
                key=lambda x: -x[1]
            )
            self._remaining_table.setRowCount(len(sorted_remaining))
            for row, (artifact, qty) in enumerate(sorted_remaining):
                self._remaining_table.setItem(row, 0, QTableWidgetItem(artifact))
                qty_item = QTableWidgetItem(f"{qty:.2f}")
                qty_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._remaining_table.setItem(row, 1, qty_item)


class SummaryWidget(QWidget):
    """
    Summary panel showing solver status and key metrics.
    Compact design - only 4-5 lines of text.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # Status
        self._status_label = QLabel("Status: Ready")
        self._status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._status_label)
        
        # Objective
        self._objective_label = QLabel("Objective: -")
        layout.addWidget(self._objective_label)
        
        # Time
        self._time_label = QLabel("Total Time: -")
        layout.addWidget(self._time_label)
        
        # Fuel
        self._fuel_label = QLabel("Fuel Usage: -")
        layout.addWidget(self._fuel_label)
        
        # Set fixed height - only needs 4 lines of text
        self.setMaximumHeight(90)
    
    def clear(self) -> None:
        """Reset to default state."""
        self._status_label.setText("Status: Ready")
        self._objective_label.setText("Objective: -")
        self._time_label.setText("Total Time: -")
        self._fuel_label.setText("Fuel Usage: -")
    
    def set_running(self) -> None:
        """Show running state."""
        self._status_label.setText("Status: Solving...")
        self._status_label.setStyleSheet("font-weight: bold; color: #f39c12;")
    
    def set_result(self, result: SolverResult, fuel_capacity: float) -> None:
        """Update from solver result."""
        # Status
        if result.status == "Optimal":
            self._status_label.setText(f"Status: {result.status}")
            self._status_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        else:
            self._status_label.setText(f"Status: {result.status}")
            self._status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        
        # Objective
        self._objective_label.setText(f"Objective: {result.objective_value:.2f}")
        
        # Time
        hours = result.total_time_hours
        if hours >= 24:
            days = hours / 24
            self._time_label.setText(f"Total Time: {hours:.1f}h ({days:.1f} days)")
        else:
            self._time_label.setText(f"Total Time: {hours:.1f} hours")
        
        # Fuel
        tank_used = result.fuel_usage.tank_total / 1e12
        remaining = fuel_capacity - tank_used
        self._fuel_label.setText(f"Fuel: {tank_used:.1f}T used, {remaining:.1f}T remaining")


class ResultsWidget(QWidget):
    """
    Combined results display widget.
    
    Contains:
    - Compact summary at top (fixed height)
    - Tabs for missions, drops, and BOM rollup (expandable)
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Summary at top (compact, fixed height)
        self._summary = SummaryWidget()
        layout.addWidget(self._summary, stretch=0)
        
        # Tabs for details (gets all remaining space)
        self._tabs = QTabWidget()
        
        # Missions tab
        self._missions_table = MissionTableWidget()
        self._tabs.addTab(self._missions_table, "Missions")
        
        # Drops tab
        self._drops_table = DropsTableWidget()
        self._tabs.addTab(self._drops_table, "Drops")
        
        # Fuel tab
        self._fuel_widget = FuelUsageWidget(compact=False)
        self._tabs.addTab(self._fuel_widget, "Fuel")
        
        # BOM Rollup tab
        self._bom_widget = BOMRollupWidget()
        self._tabs.addTab(self._bom_widget, "BOM Rollup")
        
        layout.addWidget(self._tabs, stretch=1)
    
    def clear(self) -> None:
        """Clear all results."""
        self._summary.clear()
        self._missions_table.clear()
        self._drops_table.clear()
        self._fuel_widget.clear()
        self._bom_widget.clear()
    
    def set_running(self) -> None:
        """Show solving state."""
        self._summary.set_running()
    
    def set_result(self, result: SolverResult, fuel_capacity: float) -> None:
        """Update all displays from solver result."""
        self._summary.set_result(result, fuel_capacity)
        self._missions_table.set_results(result)
        self._drops_table.set_drops(result.total_drops)
        self._fuel_widget.set_fuel_usage(result.fuel_usage)
        self._bom_widget.set_rollup(result.bom_rollup)


class PlannerResultsWidget(QWidget):
    """
    Results display for Mission Planner calculations.
    
    Similar to ResultsWidget but adapted for manual mission list calculations
    (no solver status, different summary format). Includes fuel usage breakdown.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Summary panel
        self._summary_layout = QVBoxLayout()
        self._summary_layout.setSpacing(2)
        
        self._score_label = QLabel("Score: -")
        self._score_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._summary_layout.addWidget(self._score_label)
        
        self._drops_count_label = QLabel("Total Drops: -")
        self._summary_layout.addWidget(self._drops_count_label)
        
        layout.addLayout(self._summary_layout)
        
        # Tabs for details
        self._tabs = QTabWidget()
        
        # Drops tab
        self._drops_table = DropsTableWidget()
        self._tabs.addTab(self._drops_table, "Expected Drops")
        
        # Fuel tab
        self._fuel_widget = FuelUsageWidget(compact=False)
        self._tabs.addTab(self._fuel_widget, "Fuel")
        
        # BOM Rollup tab
        self._bom_widget = BOMRollupWidget()
        self._tabs.addTab(self._bom_widget, "BOM Rollup")
        
        layout.addWidget(self._tabs, stretch=1)
    
    def clear(self) -> None:
        """Clear all results."""
        self._score_label.setText("Score: -")
        self._drops_count_label.setText("Total Drops: -")
        self._drops_table.clear()
        self._fuel_widget.clear()
        self._bom_widget.clear()
    
    def set_result(
        self,
        score: float,
        total_drops: dict[str, float],
        bom_rollup,
        fuel_by_egg: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update displays from calculation results."""
        self._score_label.setText(f"Score: {score:.2f}")
        
        total_drop_count = sum(total_drops.values())
        self._drops_count_label.setText(f"Total Drops: {total_drop_count:.1f} items")
        
        self._drops_table.set_drops(total_drops)
        
        if fuel_by_egg:
            self._fuel_widget.set_fuel_dict(fuel_by_egg)
        else:
            self._fuel_widget.clear()
        
        self._bom_widget.set_rollup(bom_rollup)


class ComparisonResultsWidget(QWidget):
    """
    Side-by-side comparison of multiple solutions.
    
    Displays 2-4 solutions in columns with diff highlighting.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        from typing import List
        from ...solution_store import SavedSolution
        
        self._solutions: List[SavedSolution] = []
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header
        self._header = QLabel("Solution Comparison")
        self._header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self._header)
        
        # Comparison table
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self._table, stretch=1)
        
        # Placeholder
        self._placeholder = QLabel("Select 2-4 solutions to compare")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #666; font-style: italic; font-size: 12px;")
        layout.addWidget(self._placeholder)
        
        self._table.hide()
    
    def clear(self) -> None:
        """Clear the comparison."""
        self._solutions = []
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._table.hide()
        self._placeholder.show()
    
    def set_solutions(self, solutions) -> None:
        """
        Set solutions to compare.
        
        Parameters
        ----------
        solutions : List[SavedSolution]
            2-4 solutions to compare side-by-side
        """
        from ...solution_store import SavedSolution
        
        if not solutions or len(solutions) < 2:
            self.clear()
            return
        
        self._solutions = solutions
        self._placeholder.hide()
        self._table.show()
        
        # Setup columns: Metric | Solution1 | Solution2 | ...
        num_solutions = len(solutions)
        self._table.setColumnCount(num_solutions + 1)
        
        headers = ["Metric"] + [s.display_name for s in solutions]
        self._table.setHorizontalHeaderLabels(headers)
        
        # Build comparison rows
        rows = [
            ("Score", [f"{s.result.objective_value:.2f}" for s in solutions]),
            ("Status", [s.result.status for s in solutions]),
            ("Total Time (h)", [f"{s.result.total_time_hours:.1f}" for s in solutions]),
            ("Mission Count", [str(len(s.mission_list)) for s in solutions]),
        ]
        
        # Find best score for highlighting
        scores = [s.result.objective_value for s in solutions]
        max_score = max(scores) if scores else 0
        
        # Add drop comparison for common artifacts
        all_drops = set()
        for s in solutions:
            all_drops.update(s.result.total_drops.keys())
        
        # Top 10 drops by max across solutions
        drop_maxes = []
        for drop in all_drops:
            max_val = max(s.result.total_drops.get(drop, 0) for s in solutions)
            drop_maxes.append((drop, max_val))
        drop_maxes.sort(key=lambda x: -x[1])
        
        for drop, _ in drop_maxes[:10]:
            values = [f"{s.result.total_drops.get(drop, 0):.2f}" for s in solutions]
            rows.append((f"Drop: {drop}", values))
        
        # Populate table
        self._table.setRowCount(len(rows))
        
        for row_idx, (metric, values) in enumerate(rows):
            # Metric column
            metric_item = QTableWidgetItem(metric)
            metric_item.setForeground(Qt.GlobalColor.darkGray)
            self._table.setItem(row_idx, 0, metric_item)
            
            # Value columns
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Highlight best score
                if metric == "Score" and solutions[col_idx].result.objective_value == max_score:
                    item.setForeground(Qt.GlobalColor.darkGreen)
                    item.setFont(item.font())
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                
                self._table.setItem(row_idx, col_idx + 1, item)

