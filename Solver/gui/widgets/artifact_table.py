"""Combined artifact table with dual-category system for ship collection and crafting targets."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QLineEdit, QPushButton, QLabel, QGroupBox, QHeaderView,
    QAbstractItemView
)

from Solver.bom import BOMEngine
from .artifact_categories import get_all_artifact_names, get_artifact_info

# Category options - same for both Ship Collection and Crafting Target
CATEGORIES = ["Targeted", "Acceptable", "Waste"]
CATEGORY_WEIGHTS = {"Targeted": 1.0, "Acceptable": 0.0, "Waste": -1.0}

# Tier ordering for display (T4 first, T1 last)
TIER_ORDER = {"T4": 0, "T3": 1, "T2": 2, "T1": 3, "": 99}


class CombinedArtifactTableWidget(QWidget):
    """
    Combined table showing all artifacts with dual-category dropdowns:
    - Ship Collection: What to target when running missions
    - Crafting Target: What to target for crafting
    
    When an artifact is set as "Targeted" for Crafting, all its ingredients
    (and their ingredients recursively) are automatically set to "Targeted"
    for Ship Collection.
    """
    
    weights_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bom = BOMEngine()
        self._setup_ui()
        self._populate_table()
        
    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Filter section
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)
        
        # Text filter
        filter_layout.addWidget(QLabel("Search:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by name...")
        self.filter_input.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.filter_input)
        
        # Craftable filter
        filter_layout.addWidget(QLabel("Type:"))
        self.craftable_filter = QComboBox()
        self.craftable_filter.addItems(["All", "Craftable Only", "Non-Craftable Only"])
        self.craftable_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.craftable_filter)
        
        # Tier filter
        filter_layout.addWidget(QLabel("Tier:"))
        self.tier_filter = QComboBox()
        self.tier_filter.addItems(["All", "T4", "T3", "T2", "T1"])
        self.tier_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.tier_filter)
        
        # Show ingredients button
        self.show_ingredients_btn = QPushButton("Show Ingredients")
        self.show_ingredients_btn.clicked.connect(self._show_ingredients)
        self.show_ingredients_btn.setToolTip("Filter to show selected artifact and all its crafting ingredients")
        filter_layout.addWidget(self.show_ingredients_btn)
        
        # Clear filter button
        self.clear_filter_btn = QPushButton("Clear Filters")
        self.clear_filter_btn.clicked.connect(self._clear_filters)
        filter_layout.addWidget(self.clear_filter_btn)
        
        filter_layout.addStretch()
        layout.addWidget(filter_group)
        
        # Bulk actions section
        bulk_group = QGroupBox("Bulk Actions")
        bulk_layout = QHBoxLayout(bulk_group)
        
        # Ship Collection bulk action
        bulk_layout.addWidget(QLabel("Set Visible Ship Collection:"))
        self.bulk_ship_combo = QComboBox()
        self.bulk_ship_combo.addItems(CATEGORIES)
        bulk_layout.addWidget(self.bulk_ship_combo)
        
        self.apply_ship_btn = QPushButton("Apply to Visible")
        self.apply_ship_btn.clicked.connect(self._apply_bulk_ship)
        bulk_layout.addWidget(self.apply_ship_btn)
        
        bulk_layout.addSpacing(20)
        
        # Crafting Target bulk action
        bulk_layout.addWidget(QLabel("Set Visible Crafting Target:"))
        self.bulk_craft_combo = QComboBox()
        self.bulk_craft_combo.addItems(CATEGORIES)
        bulk_layout.addWidget(self.bulk_craft_combo)
        
        self.apply_craft_btn = QPushButton("Apply to Visible")
        self.apply_craft_btn.clicked.connect(self._apply_bulk_craft)
        bulk_layout.addWidget(self.apply_craft_btn)
        
        bulk_layout.addStretch()
        layout.addWidget(bulk_group)
        
        # Main table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Family", "Tier", "Artifact", "Craftable", "Ship Collection", "Crafting Target"
        ])
        
        # Configure table
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(False)  # We handle sorting ourselves
        
        # Set column resize modes
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Family
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Tier
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Artifact
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Craftable
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Ship Collection
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Crafting Target
        
        layout.addWidget(self.table)
        
        # Status bar
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
    def _get_sorted_artifacts(self) -> list[tuple[str, str, str, bool]]:
        """
        Get all artifacts sorted by family name, then tier (T4→T1).
        Returns list of (artifact_name, family, tier, is_craftable).
        """
        artifact_info = get_artifact_info()
        artifacts = []
        
        for name, info in artifact_info.items():
            family = info.get("family", "Unknown")
            tier_num = info.get("tier_number", 0)
            tier = f"T{tier_num}" if tier_num > 0 else ""
            is_craftable = info.get("craftable", False)
            artifacts.append((name, family, tier, is_craftable))
        
        # Sort by family name, then by tier (T4 first)
        artifacts.sort(key=lambda x: (x[1], TIER_ORDER.get(x[2], 99)))
        
        return artifacts
    
    def _populate_table(self):
        """Populate the table with all artifacts."""
        artifacts = self._get_sorted_artifacts()
        
        self.table.setRowCount(len(artifacts))
        self._row_data = {}  # Store artifact name for each row
        
        for row, (artifact, family, tier, is_craftable) in enumerate(artifacts):
            self._row_data[row] = artifact
            
            # Family
            family_item = QTableWidgetItem(family)
            family_item.setFlags(family_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, family_item)
            
            # Tier
            tier_item = QTableWidgetItem(tier)
            tier_item.setFlags(tier_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            tier_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, tier_item)
            
            # Artifact name
            name_item = QTableWidgetItem(artifact)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, name_item)
            
            # Craftable indicator
            craft_item = QTableWidgetItem("✓" if is_craftable else "")
            craft_item.setFlags(craft_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            craft_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, craft_item)
            
            # Ship Collection dropdown
            ship_combo = QComboBox()
            ship_combo.addItems(CATEGORIES)
            ship_combo.setCurrentText("Acceptable")  # Default
            ship_combo.setProperty("artifact", artifact)
            ship_combo.setProperty("column", "ship")
            ship_combo.currentTextChanged.connect(self._on_category_changed)
            self.table.setCellWidget(row, 4, ship_combo)
            
            # Crafting Target dropdown
            craft_combo = QComboBox()
            craft_combo.addItems(CATEGORIES)
            craft_combo.setCurrentText("Acceptable")  # Default
            craft_combo.setProperty("artifact", artifact)
            craft_combo.setProperty("column", "craft")
            craft_combo.currentTextChanged.connect(self._on_crafting_target_changed)
            self.table.setCellWidget(row, 5, craft_combo)
        
        self._update_status()
    
    def _on_category_changed(self, value: str):
        """Handle ship collection category change."""
        self.weights_changed.emit()
    
    def _on_crafting_target_changed(self, value: str):
        """
        Handle crafting target change.
        If set to "Targeted", automatically set all ingredients to "Targeted" for ship collection.
        """
        combo = self.sender()
        artifact = combo.property("artifact")
        
        if value == "Targeted":
            # Get all ingredients for this artifact (returns dict of artifact_name -> base_equivalence)
            contributors = self.bom.get_all_contributors_for_target(artifact)
            ingredient_names = set(contributors.keys())
            
            # Set all ingredients to "Targeted" for ship collection
            for row in range(self.table.rowCount()):
                row_artifact = self._row_data.get(row)
                if row_artifact in ingredient_names:
                    ship_combo = self.table.cellWidget(row, 4)
                    if ship_combo:
                        # Only upgrade to Targeted, don't downgrade
                        if ship_combo.currentText() != "Targeted":
                            ship_combo.blockSignals(True)
                            ship_combo.setCurrentText("Targeted")
                            ship_combo.blockSignals(False)
        
        self.weights_changed.emit()
    
    def _apply_filters(self):
        """Apply current filter settings to show/hide rows."""
        search_text = self.filter_input.text().lower()
        craftable_filter = self.craftable_filter.currentText()
        tier_filter = self.tier_filter.currentText()
        
        visible_count = 0
        
        for row in range(self.table.rowCount()):
            artifact = self._row_data.get(row, "")
            tier = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
            is_craftable = self.table.item(row, 3).text() == "✓"
            
            # Apply text filter
            text_match = search_text == "" or search_text in artifact.lower()
            
            # Apply craftable filter
            if craftable_filter == "Craftable Only":
                craftable_match = is_craftable
            elif craftable_filter == "Non-Craftable Only":
                craftable_match = not is_craftable
            else:
                craftable_match = True
            
            # Apply tier filter
            tier_match = tier_filter == "All" or tier == tier_filter
            
            # Show/hide row
            visible = text_match and craftable_match and tier_match
            self.table.setRowHidden(row, not visible)
            
            if visible:
                visible_count += 1
        
        self._update_status(visible_count)
    
    def _clear_filters(self):
        """Clear all filters."""
        self.filter_input.clear()
        self.craftable_filter.setCurrentIndex(0)
        self.tier_filter.setCurrentIndex(0)
    
    def _show_ingredients(self):
        """Show the selected artifact and all its ingredients."""
        selected = self.table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        artifact = self._row_data.get(row)
        if not artifact:
            return
        
        # Get all ingredients (returns dict of artifact_name -> base_equivalence)
        contributors = self.bom.get_all_contributors_for_target(artifact)
        ingredient_names = set(contributors.keys())
        ingredient_names.add(artifact)  # Include the artifact itself
        
        # Clear other filters
        self.craftable_filter.setCurrentIndex(0)
        self.tier_filter.setCurrentIndex(0)
        
        # Set search to show only these artifacts
        visible_count = 0
        for r in range(self.table.rowCount()):
            row_artifact = self._row_data.get(r, "")
            visible = row_artifact in ingredient_names
            self.table.setRowHidden(r, not visible)
            if visible:
                visible_count += 1
        
        # Update filter input to indicate special filter (block signals to prevent _apply_filters)
        self.filter_input.blockSignals(True)
        self.filter_input.setText(f"[Ingredients of {artifact}]")
        self.filter_input.blockSignals(False)
        self._update_status(visible_count)
    
    def _apply_bulk_ship(self):
        """Apply bulk category to visible rows for Ship Collection."""
        category = self.bulk_ship_combo.currentText()
        
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                combo = self.table.cellWidget(row, 4)
                if combo:
                    combo.setCurrentText(category)
        
        self.weights_changed.emit()
    
    def _apply_bulk_craft(self):
        """Apply bulk category to visible rows for Crafting Target."""
        category = self.bulk_craft_combo.currentText()
        
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                combo = self.table.cellWidget(row, 5)
                if combo:
                    combo.setCurrentText(category)
        
        # Note: _on_crafting_target_changed will handle ingredient cascading
        self.weights_changed.emit()
    
    def _update_status(self, visible_count: int = None):
        """Update the status label."""
        total = self.table.rowCount()
        if visible_count is None:
            visible_count = sum(1 for row in range(total) if not self.table.isRowHidden(row))
        
        # Count targeted items
        ship_targeted = 0
        craft_targeted = 0
        
        for row in range(total):
            ship_combo = self.table.cellWidget(row, 4)
            craft_combo = self.table.cellWidget(row, 5)
            
            if ship_combo and ship_combo.currentText() == "Targeted":
                ship_targeted += 1
            if craft_combo and craft_combo.currentText() == "Targeted":
                craft_targeted += 1
        
        self.status_label.setText(
            f"Showing {visible_count} of {total} artifacts | "
            f"Ship Targeted: {ship_targeted} | Craft Targeted: {craft_targeted}"
        )
    
    def get_weights(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Get the weight dictionaries for both ship collection and crafting.
        
        Returns:
            Tuple of (ship_weights, craft_weights) where each is a dict
            mapping artifact name to weight value.
        """
        ship_weights = {}
        craft_weights = {}
        
        for row in range(self.table.rowCount()):
            artifact = self._row_data.get(row)
            if not artifact:
                continue
            
            ship_combo = self.table.cellWidget(row, 4)
            craft_combo = self.table.cellWidget(row, 5)
            
            if ship_combo:
                ship_weights[artifact] = CATEGORY_WEIGHTS[ship_combo.currentText()]
            if craft_combo:
                craft_weights[artifact] = CATEGORY_WEIGHTS[craft_combo.currentText()]
        
        return ship_weights, craft_weights
    
    def set_weights(self, ship_weights: dict[str, float], craft_weights: dict[str, float] = None):
        """
        Set the category selections from weight dictionaries.
        
        Args:
            ship_weights: Dict mapping artifact name to weight for ship collection
            craft_weights: Dict mapping artifact name to weight for crafting (optional)
        """
        if craft_weights is None:
            craft_weights = {}
        
        # Reverse lookup for weights to categories
        weight_to_category = {v: k for k, v in CATEGORY_WEIGHTS.items()}
        
        for row in range(self.table.rowCount()):
            artifact = self._row_data.get(row)
            if not artifact:
                continue
            
            # Set ship collection
            if artifact in ship_weights:
                ship_combo = self.table.cellWidget(row, 4)
                if ship_combo:
                    weight = ship_weights[artifact]
                    category = weight_to_category.get(weight, "Acceptable")
                    ship_combo.blockSignals(True)
                    ship_combo.setCurrentText(category)
                    ship_combo.blockSignals(False)
            
            # Set crafting target
            if artifact in craft_weights:
                craft_combo = self.table.cellWidget(row, 5)
                if craft_combo:
                    weight = craft_weights[artifact]
                    category = weight_to_category.get(weight, "Acceptable")
                    craft_combo.blockSignals(True)
                    craft_combo.setCurrentText(category)
                    craft_combo.blockSignals(False)
        
        self._update_status()
        self.weights_changed.emit()
