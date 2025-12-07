"""
Mission list editor widget for the Mission Planner.

Allows users to manually build a list of missions with ship, duration,
level, target artifact, and count selections. Shows running fuel totals.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QComboBox,
    QSpinBox,
    QLabel,
    QGroupBox,
    QMessageBox,
    QAbstractItemView,
    QFrame,
)

from ...config import SHIP_METADATA
from ...mission_data import (
    build_mission_inventory,
    get_available_targets,
    get_available_durations,
    get_available_levels,
    get_fuel_requirements,
    MissionOption,
)
from ...solution_store import MissionListItem
from ...aliases import get_egg_display_name
from .results import FuelUsageWidget, format_fuel_amount


class MissionListWidget(QWidget):
    """
    Widget for editing a list of missions.
    
    Provides controls to add missions with ship/duration/level/target selection,
    displays them in a table, and emits signals when the list changes.
    
    Signals
    -------
    list_changed : Signal
        Emitted when the mission list is modified
    calculate_requested : Signal
        Emitted when user requests calculation
    """
    
    list_changed = Signal()
    calculate_requested = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Cache mission inventory for dropdown population
        self._inventory: Optional[List[MissionOption]] = None
        self._loading_inventory = False
        
        self._setup_ui()
        self._connect_signals()
        
        # Load inventory in background
        self._load_inventory()
    
    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Add mission controls
        add_group = QGroupBox("Add Mission")
        add_layout = QHBoxLayout(add_group)
        add_layout.setSpacing(8)
        
        # Ship selector
        ship_label = QLabel("Ship:")
        self._ship_combo = QComboBox()
        self._ship_combo.setMinimumWidth(120)
        # Populate with ships from metadata (excluding CHICKEN_ONE tutorial ship)
        for api_name, (display_name, max_level) in SHIP_METADATA.items():
            if api_name != "CHICKEN_ONE":
                self._ship_combo.addItem(display_name, api_name)
        # Default to Henerprise
        henerprise_idx = self._ship_combo.findData("HENERPRISE")
        if henerprise_idx >= 0:
            self._ship_combo.setCurrentIndex(henerprise_idx)
        
        add_layout.addWidget(ship_label)
        add_layout.addWidget(self._ship_combo)
        
        # Duration selector
        duration_label = QLabel("Duration:")
        self._duration_combo = QComboBox()
        self._duration_combo.setMinimumWidth(80)
        self._duration_combo.addItems(["Short", "Long", "Epic"])
        
        add_layout.addWidget(duration_label)
        add_layout.addWidget(self._duration_combo)
        
        # Level selector
        level_label = QLabel("Level:")
        self._level_spin = QSpinBox()
        self._level_spin.setMinimum(0)
        self._level_spin.setMaximum(8)
        self._level_spin.setValue(8)  # Default to max
        
        add_layout.addWidget(level_label)
        add_layout.addWidget(self._level_spin)
        
        # Target selector
        target_label = QLabel("Target:")
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(150)
        self._target_combo.addItem("Any", None)
        
        add_layout.addWidget(target_label)
        add_layout.addWidget(self._target_combo)
        
        # Count selector
        count_label = QLabel("Count:")
        self._count_spin = QSpinBox()
        self._count_spin.setMinimum(1)
        self._count_spin.setMaximum(1000)
        self._count_spin.setValue(1)
        
        add_layout.addWidget(count_label)
        add_layout.addWidget(self._count_spin)
        
        # Add button
        self._add_btn = QPushButton("Add")
        self._add_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
        """)
        add_layout.addWidget(self._add_btn)
        
        add_layout.addStretch()
        layout.addWidget(add_group)
        
        # Mission list table
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "Ship", "Duration", "Level", "Target", "Count", ""
        ])
        
        # Configure columns
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        layout.addWidget(self._table, stretch=1)
        
        # Fuel usage display (running total)
        fuel_group = QGroupBox("Running Fuel Total")
        fuel_layout = QVBoxLayout(fuel_group)
        fuel_layout.setContentsMargins(8, 8, 8, 8)
        self._fuel_display = FuelUsageWidget(compact=False)
        fuel_layout.addWidget(self._fuel_display)
        layout.addWidget(fuel_group)
        
        # Bottom button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._on_clear_all)
        btn_row.addWidget(self._clear_btn)
        
        btn_row.addStretch()
        
        self._calc_btn = QPushButton("Calculate")
        self._calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        btn_row.addWidget(self._calc_btn)
        
        layout.addLayout(btn_row)
    
    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._ship_combo.currentIndexChanged.connect(self._on_ship_changed)
        self._duration_combo.currentIndexChanged.connect(self._on_duration_changed)
        self._level_spin.valueChanged.connect(self._on_level_changed)
        self._add_btn.clicked.connect(self._on_add_mission)
        self._calc_btn.clicked.connect(self.calculate_requested.emit)
        # Update fuel display when list changes
        self.list_changed.connect(self._update_fuel_display)
    
    def _update_fuel_display(self) -> None:
        """Update the running fuel total display."""
        fuel_totals = self._calculate_fuel_totals()
        self._fuel_display.set_fuel_dict(fuel_totals)
    
    def _calculate_fuel_totals(self) -> Dict[str, float]:
        """Calculate total fuel usage from current mission list."""
        totals: Dict[str, float] = {}
        
        for row in range(self._table.rowCount()):
            ship_item = self._table.item(row, 0)
            duration_item = self._table.item(row, 1)
            count_item = self._table.item(row, 4)
            
            if not all([ship_item, duration_item, count_item]):
                continue
            
            ship = ship_item.data(Qt.ItemDataRole.UserRole)
            duration = duration_item.data(Qt.ItemDataRole.UserRole)
            count = int(count_item.text())
            
            fuel_reqs = get_fuel_requirements(ship, duration)
            for egg, amount in fuel_reqs.items():
                totals[egg] = totals.get(egg, 0) + amount * count
        
        return totals
    
    def get_fuel_totals(self) -> Dict[str, float]:
        """
        Get the current fuel totals for the mission list.
        
        Returns
        -------
        Dict[str, float]
            Mapping of egg name to total amount needed
        """
        return self._calculate_fuel_totals()
    
    def _load_inventory(self) -> None:
        """Load mission inventory (can be slow, ideally done async)."""
        if self._inventory is not None or self._loading_inventory:
            return
        
        self._loading_inventory = True
        try:
            self._inventory = build_mission_inventory()
            self._update_target_dropdown()
        finally:
            self._loading_inventory = False
    
    def _on_ship_changed(self) -> None:
        """Handle ship selection change - update duration and level options."""
        self._update_duration_dropdown()
        self._update_level_limits()
        self._update_target_dropdown()
    
    def _on_duration_changed(self) -> None:
        """Handle duration change - update level and target options."""
        self._update_level_limits()
        self._update_target_dropdown()
    
    def _on_level_changed(self) -> None:
        """Handle level change - update target options."""
        self._update_target_dropdown()
    
    def _update_duration_dropdown(self) -> None:
        """Update duration dropdown based on selected ship."""
        if self._inventory is None:
            return
        
        ship_api = self._ship_combo.currentData()
        if not ship_api:
            return
        
        durations = get_available_durations(ship_api, self._inventory)
        
        current = self._duration_combo.currentText().upper()
        self._duration_combo.blockSignals(True)
        self._duration_combo.clear()
        
        for dur in durations:
            self._duration_combo.addItem(dur.capitalize(), dur)
        
        # Restore selection if possible
        idx = self._duration_combo.findData(current)
        if idx >= 0:
            self._duration_combo.setCurrentIndex(idx)
        
        self._duration_combo.blockSignals(False)
    
    def _update_level_limits(self) -> None:
        """Update level spinner based on ship metadata."""
        ship_api = self._ship_combo.currentData()
        if not ship_api:
            return
        
        _, max_level = SHIP_METADATA.get(ship_api, ("", 0))
        
        self._level_spin.blockSignals(True)
        self._level_spin.setMaximum(max_level)
        if self._level_spin.value() > max_level:
            self._level_spin.setValue(max_level)
        self._level_spin.blockSignals(False)
    
    def _update_target_dropdown(self) -> None:
        """Update target dropdown based on ship/duration/level selection."""
        if self._inventory is None:
            return
        
        ship_api = self._ship_combo.currentData()
        duration = self._duration_combo.currentData() or self._duration_combo.currentText().upper()
        level = self._level_spin.value()
        
        if not ship_api:
            return
        
        targets = get_available_targets(ship_api, duration, level, self._inventory)
        
        current = self._target_combo.currentText()
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        
        for target in targets:
            data = None if target == "Any" else target
            self._target_combo.addItem(target, data)
        
        # Restore selection if possible
        idx = self._target_combo.findText(current)
        if idx >= 0:
            self._target_combo.setCurrentIndex(idx)
        
        self._target_combo.blockSignals(False)
    
    def _on_add_mission(self) -> None:
        """Add a mission to the list."""
        ship_api = self._ship_combo.currentData()
        ship_label = self._ship_combo.currentText()
        duration = self._duration_combo.currentData() or self._duration_combo.currentText().upper()
        level = self._level_spin.value()
        target = self._target_combo.currentData()  # None for "Any"
        count = self._count_spin.value()
        
        if not ship_api:
            return
        
        self._add_table_row(ship_api, ship_label, duration, level, target, count)
        self.list_changed.emit()
    
    def _add_table_row(
        self,
        ship_api: str,
        ship_label: str,
        duration: str,
        level: int,
        target: Optional[str],
        count: int,
    ) -> None:
        """Add a row to the table."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        
        # Ship
        ship_item = QTableWidgetItem(ship_label)
        ship_item.setData(Qt.ItemDataRole.UserRole, ship_api)
        self._table.setItem(row, 0, ship_item)
        
        # Duration
        duration_item = QTableWidgetItem(duration.capitalize())
        duration_item.setData(Qt.ItemDataRole.UserRole, duration.upper())
        self._table.setItem(row, 1, duration_item)
        
        # Level
        level_item = QTableWidgetItem(str(level))
        level_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 2, level_item)
        
        # Target
        target_display = target if target else "Any"
        target_item = QTableWidgetItem(target_display)
        target_item.setData(Qt.ItemDataRole.UserRole, target)
        self._table.setItem(row, 3, target_item)
        
        # Count
        count_item = QTableWidgetItem(str(count))
        count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 4, count_item)
        
        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        remove_btn.clicked.connect(lambda: self._remove_row(row))
        self._table.setCellWidget(row, 5, remove_btn)
    
    def _remove_row(self, row: int) -> None:
        """Remove a row from the table."""
        # Find the actual row (buttons may have shifted)
        sender = self.sender()
        for r in range(self._table.rowCount()):
            if self._table.cellWidget(r, 5) == sender:
                self._table.removeRow(r)
                self.list_changed.emit()
                return
    
    def _on_clear_all(self) -> None:
        """Clear all missions from the list."""
        if self._table.rowCount() == 0:
            return
        
        reply = QMessageBox.question(
            self,
            "Clear Mission List",
            "Are you sure you want to clear all missions?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._table.setRowCount(0)
            self.list_changed.emit()
    
    def get_mission_list(self) -> List[MissionListItem]:
        """
        Get the current mission list.
        
        Returns
        -------
        List[MissionListItem]
            List of missions
        """
        missions = []
        
        for row in range(self._table.rowCount()):
            ship_item = self._table.item(row, 0)
            duration_item = self._table.item(row, 1)
            level_item = self._table.item(row, 2)
            target_item = self._table.item(row, 3)
            count_item = self._table.item(row, 4)
            
            if not all([ship_item, duration_item, level_item, target_item, count_item]):
                continue
            
            missions.append(MissionListItem(
                ship=ship_item.data(Qt.ItemDataRole.UserRole),
                ship_label=ship_item.text(),
                duration=duration_item.data(Qt.ItemDataRole.UserRole),
                level=int(level_item.text()),
                target=target_item.data(Qt.ItemDataRole.UserRole),
                count=int(count_item.text()),
            ))
        
        return missions
    
    def get_mission_tuples(self) -> List[Tuple[str, str, Optional[str], int]]:
        """
        Get mission list as tuples for BOM calculation.
        
        Rolls up identical missions (same ship/duration/target) by summing counts.
        
        Returns
        -------
        List[Tuple[str, str, Optional[str], int]]
            List of (ship, duration, target, count) tuples
        """
        missions = self.get_mission_list()
        
        # Roll up identical missions
        rolled_up: Dict[Tuple[str, str, Optional[str]], int] = {}
        
        for m in missions:
            key = (m.ship, m.duration, m.target)
            rolled_up[key] = rolled_up.get(key, 0) + m.count
        
        return [
            (ship, duration, target, count)
            for (ship, duration, target), count in rolled_up.items()
        ]
    
    def set_mission_list(self, missions: List[MissionListItem]) -> None:
        """
        Set the mission list from a list of items.
        
        Parameters
        ----------
        missions : List[MissionListItem]
            Missions to load
        """
        self._table.setRowCount(0)
        
        for m in missions:
            self._add_table_row(
                m.ship,
                m.ship_label,
                m.duration,
                m.level,
                m.target,
                m.count,
            )
        
        self.list_changed.emit()
    
    def clear(self) -> None:
        """Clear the mission list without confirmation."""
        self._table.setRowCount(0)
        self.list_changed.emit()
