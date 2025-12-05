"""
Artifact categorization widget with filter and bulk actions.

Provides a streamlined interface for categorizing artifacts as:
- Targeted: Artifacts to optimize for (weight = 1.0)
- Acceptable: Neutral items, neither prioritized nor penalized (weight = 0.0)  
- Waste: Unwanted drops (weight = -1.0)
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QComboBox,
    QPushButton,
    QFrame,
    QAbstractItemView,
    QSplitter,
    QGroupBox,
    QCheckBox,
)

from ...config import UserConfig


# Data paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WASMEGG_DIR = BASE_DIR.parent / "Wasmegg"
EIAFX_DATA_PATH = WASMEGG_DIR / "eiafx-data.json"


class ArtifactCategory(Enum):
    """Categories for artifact classification."""
    TARGETED = "targeted"      # Weight = 1.0, actively sought
    ACCEPTABLE = "acceptable"  # Weight = 0.0, neutral
    WASTE = "waste"           # Weight = -1.0, avoid


CATEGORY_WEIGHTS = {
    ArtifactCategory.TARGETED: 1.0,
    ArtifactCategory.ACCEPTABLE: 0.0,
    ArtifactCategory.WASTE: -1.0,
}

CATEGORY_COLORS = {
    ArtifactCategory.TARGETED: "#2e7d32",     # Green
    ArtifactCategory.ACCEPTABLE: "#757575",   # Gray
    ArtifactCategory.WASTE: "#c62828",        # Red
}

CATEGORY_LABELS = {
    ArtifactCategory.TARGETED: "✓ Targeted",
    ArtifactCategory.ACCEPTABLE: "○ Acceptable", 
    ArtifactCategory.WASTE: "✗ Waste",
}


def load_artifact_data() -> Dict[str, List[Dict]]:
    """
    Load artifact data from eiafx-data.json.
    
    Returns a dict mapping family name to list of tier info dicts.
    Each tier has 'name', 'tier_number', 'craftable' keys.
    """
    if not EIAFX_DATA_PATH.exists():
        return {}
    
    try:
        with EIAFX_DATA_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
    
    result = {}
    for family in data.get("artifact_families", []):
        family_name = family.get("name", "Unknown")
        tiers = []
        for tier in family.get("tiers", []):
            tier_info = {
                "name": tier.get("name", ""),
                "tier_number": tier.get("tier_number", 0),
                "craftable": bool(tier.get("recipe")),
                "family": family_name,
            }
            tiers.append(tier_info)
        result[family_name] = tiers
    
    return result


def get_all_artifact_names() -> List[str]:
    """Get a sorted list of all artifact display names."""
    data = load_artifact_data()
    names = []
    for family, tiers in data.items():
        for tier in tiers:
            names.append(tier["name"])
    return sorted(names)


def get_artifact_info() -> Dict[str, Dict]:
    """
    Get artifact metadata keyed by name.
    
    Returns dict mapping artifact name to {family, tier_number, craftable}.
    """
    data = load_artifact_data()
    info = {}
    for family, tiers in data.items():
        for tier in tiers:
            info[tier["name"]] = {
                "family": family,
                "tier_number": tier["tier_number"],
                "craftable": tier["craftable"],
            }
    return info


class ArtifactListItem(QListWidgetItem):
    """List item with artifact metadata."""
    
    def __init__(self, name: str, info: Dict, category: ArtifactCategory):
        super().__init__()
        self.artifact_name = name
        self.artifact_info = info
        self._category = category
        self._update_display()
    
    @property
    def category(self) -> ArtifactCategory:
        return self._category
    
    @category.setter
    def category(self, value: ArtifactCategory):
        self._category = value
        self._update_display()
    
    def _update_display(self):
        """Update the display text and styling."""
        tier = self.artifact_info.get("tier_number", 0)
        tier_label = f"T{tier}" if tier > 0 else ""
        craftable = "⚒" if self.artifact_info.get("craftable") else ""
        
        # Format: "Name [T3 ⚒] - ✓ Targeted"
        category_label = CATEGORY_LABELS[self._category]
        display = f"{self.artifact_name}"
        if tier_label or craftable:
            display += f" [{tier_label}{craftable}]"
        display += f"  —  {category_label}"
        
        self.setText(display)
        
        # Set color based on category
        color = QColor(CATEGORY_COLORS[self._category])
        self.setForeground(color)


class ArtifactCategoryWidget(QWidget):
    """
    Widget for categorizing artifacts with filter and bulk actions.
    
    Features:
    - Searchable/filterable artifact list
    - Filter by family, tier, or current category
    - Multi-select with Shift/Ctrl-click
    - Bulk action buttons to set category for selection
    - Preset buttons for common categorization patterns
    
    Signals:
        categories_changed(dict): Emitted when any category changes.
            Dict maps artifact name -> weight (1.0, 0.0, or -1.0)
    """
    
    categories_changed = Signal(dict)
    
    def __init__(
        self,
        title: str = "Artifact Categories",
        initial_weights: Optional[Dict[str, float]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        
        self._artifact_info = get_artifact_info()
        self._all_names = sorted(self._artifact_info.keys())
        
        # Categories keyed by artifact name
        self._categories: Dict[str, ArtifactCategory] = {}
        
        # Initialize categories from weights or default to Acceptable
        if initial_weights:
            self._load_from_weights(initial_weights)
        else:
            for name in self._all_names:
                self._categories[name] = ArtifactCategory.ACCEPTABLE
        
        self._setup_ui(title)
        self._populate_list()
    
    def _load_from_weights(self, weights: Dict[str, float]) -> None:
        """Load categories from weight dict."""
        for name in self._all_names:
            weight = weights.get(name, 0.0)
            if weight > 0.5:
                self._categories[name] = ArtifactCategory.TARGETED
            elif weight < -0.5:
                self._categories[name] = ArtifactCategory.WASTE
            else:
                self._categories[name] = ArtifactCategory.ACCEPTABLE
    
    def _setup_ui(self, title: str) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title_label)
        
        # Description
        desc = QLabel(
            "Categorize artifacts: Targeted (collect), Acceptable (neutral), Waste (avoid). "
            "Use filters and bulk actions to quickly categorize many items."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 4px;")
        layout.addWidget(desc)
        
        # Filter row
        filter_frame = QFrame()
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(8)
        
        # Search box
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("🔍 Search artifacts...")
        self._search_box.setClearButtonEnabled(True)
        self._search_box.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._search_box, stretch=2)
        
        # Family filter
        self._family_filter = QComboBox()
        self._family_filter.addItem("All Families")
        families = sorted(set(info["family"] for info in self._artifact_info.values()))
        self._family_filter.addItems(families)
        self._family_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._family_filter)
        
        # Tier filter
        self._tier_filter = QComboBox()
        self._tier_filter.addItem("All Tiers")
        self._tier_filter.addItems(["T1", "T2", "T3", "T4"])
        self._tier_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._tier_filter)
        
        # Category filter
        self._category_filter = QComboBox()
        self._category_filter.addItem("All Categories")
        for cat in ArtifactCategory:
            self._category_filter.addItem(CATEGORY_LABELS[cat])
        self._category_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self._category_filter)
        
        layout.addWidget(filter_frame)
        
        # Artifact list
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.setAlternatingRowColors(True)
        self._list.setStyleSheet("""
            QListWidget::item { padding: 4px 8px; }
            QListWidget::item:selected { background-color: #1976d2; color: white; }
        """)
        layout.addWidget(self._list, stretch=1)
        
        # Selection info
        self._selection_label = QLabel("0 selected")
        self._selection_label.setStyleSheet("color: #666; font-size: 11px;")
        self._list.itemSelectionChanged.connect(self._update_selection_label)
        layout.addWidget(self._selection_label)
        
        # Bulk action buttons
        action_frame = QFrame()
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        
        action_label = QLabel("Set selected to:")
        action_layout.addWidget(action_label)
        
        # Category buttons
        self._btn_targeted = QPushButton("✓ Targeted")
        self._btn_targeted.setStyleSheet(f"color: {CATEGORY_COLORS[ArtifactCategory.TARGETED]};")
        self._btn_targeted.clicked.connect(lambda: self._set_selected_category(ArtifactCategory.TARGETED))
        action_layout.addWidget(self._btn_targeted)
        
        self._btn_acceptable = QPushButton("○ Acceptable")
        self._btn_acceptable.setStyleSheet(f"color: {CATEGORY_COLORS[ArtifactCategory.ACCEPTABLE]};")
        self._btn_acceptable.clicked.connect(lambda: self._set_selected_category(ArtifactCategory.ACCEPTABLE))
        action_layout.addWidget(self._btn_acceptable)
        
        self._btn_waste = QPushButton("✗ Waste")
        self._btn_waste.setStyleSheet(f"color: {CATEGORY_COLORS[ArtifactCategory.WASTE]};")
        self._btn_waste.clicked.connect(lambda: self._set_selected_category(ArtifactCategory.WASTE))
        action_layout.addWidget(self._btn_waste)
        
        action_layout.addStretch()
        layout.addWidget(action_frame)
        
        # Preset buttons
        preset_frame = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout(preset_frame)
        preset_layout.setSpacing(8)
        
        btn_target_t4 = QPushButton("Target All T4")
        btn_target_t4.clicked.connect(self._preset_target_t4)
        preset_layout.addWidget(btn_target_t4)
        
        btn_target_craftable = QPushButton("Target Craftable")
        btn_target_craftable.clicked.connect(self._preset_target_craftable)
        preset_layout.addWidget(btn_target_craftable)
        
        btn_waste_t1 = QPushButton("Waste T1")
        btn_waste_t1.clicked.connect(self._preset_waste_t1)
        preset_layout.addWidget(btn_waste_t1)
        
        btn_all_acceptable = QPushButton("All Acceptable")
        btn_all_acceptable.clicked.connect(self._preset_all_acceptable)
        preset_layout.addWidget(btn_all_acceptable)
        
        preset_layout.addStretch()
        layout.addWidget(preset_frame)
    
    def _populate_list(self) -> None:
        """Populate the list with all artifacts."""
        self._list.clear()
        self._items: Dict[str, ArtifactListItem] = {}
        
        for name in self._all_names:
            info = self._artifact_info.get(name, {})
            category = self._categories.get(name, ArtifactCategory.ACCEPTABLE)
            item = ArtifactListItem(name, info, category)
            self._items[name] = item
            self._list.addItem(item)
    
    def _apply_filters(self) -> None:
        """Apply current filters to the list."""
        search_text = self._search_box.text().lower()
        family_filter = self._family_filter.currentText()
        tier_filter = self._tier_filter.currentText()
        category_filter = self._category_filter.currentText()
        
        for name, item in self._items.items():
            visible = True
            info = self._artifact_info.get(name, {})
            
            # Search filter
            if search_text and search_text not in name.lower():
                visible = False
            
            # Family filter
            if visible and family_filter != "All Families":
                if info.get("family") != family_filter:
                    visible = False
            
            # Tier filter
            if visible and tier_filter != "All Tiers":
                tier_num = int(tier_filter[1])  # "T3" -> 3
                if info.get("tier_number") != tier_num:
                    visible = False
            
            # Category filter
            if visible and category_filter != "All Categories":
                cat_label = CATEGORY_LABELS[item.category]
                if cat_label != category_filter:
                    visible = False
            
            item.setHidden(not visible)
    
    def _update_selection_label(self) -> None:
        """Update the selection count label."""
        count = len(self._list.selectedItems())
        visible_count = sum(1 for item in self._items.values() if not item.isHidden())
        self._selection_label.setText(f"{count} selected of {visible_count} visible")
    
    def _set_selected_category(self, category: ArtifactCategory) -> None:
        """Set the category for all selected items."""
        for item in self._list.selectedItems():
            if isinstance(item, ArtifactListItem):
                item.category = category
                self._categories[item.artifact_name] = category
        
        self._emit_changes()
    
    def _preset_target_t4(self) -> None:
        """Preset: Target all T4 artifacts."""
        for name, item in self._items.items():
            info = self._artifact_info.get(name, {})
            if info.get("tier_number") == 4:
                item.category = ArtifactCategory.TARGETED
                self._categories[name] = ArtifactCategory.TARGETED
        self._emit_changes()
    
    def _preset_target_craftable(self) -> None:
        """Preset: Target all craftable artifacts."""
        for name, item in self._items.items():
            info = self._artifact_info.get(name, {})
            if info.get("craftable"):
                item.category = ArtifactCategory.TARGETED
                self._categories[name] = ArtifactCategory.TARGETED
        self._emit_changes()
    
    def _preset_waste_t1(self) -> None:
        """Preset: Mark all T1 as Waste."""
        for name, item in self._items.items():
            info = self._artifact_info.get(name, {})
            if info.get("tier_number") == 1:
                item.category = ArtifactCategory.WASTE
                self._categories[name] = ArtifactCategory.WASTE
        self._emit_changes()
    
    def _preset_all_acceptable(self) -> None:
        """Preset: Set all to Acceptable."""
        for name, item in self._items.items():
            item.category = ArtifactCategory.ACCEPTABLE
            self._categories[name] = ArtifactCategory.ACCEPTABLE
        self._emit_changes()
    
    def _emit_changes(self) -> None:
        """Emit the categories_changed signal with current weights."""
        weights = self.get_weights()
        self.categories_changed.emit(weights)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights dict (artifact name -> weight value)."""
        return {
            name: CATEGORY_WEIGHTS[cat]
            for name, cat in self._categories.items()
        }
    
    def get_categories(self) -> Dict[str, ArtifactCategory]:
        """Get current categories dict."""
        return dict(self._categories)
    
    def update_from_weights(self, weights: Dict[str, float]) -> None:
        """Update categories from a weights dict."""
        self._load_from_weights(weights)
        # Update list items
        for name, item in self._items.items():
            item.category = self._categories.get(name, ArtifactCategory.ACCEPTABLE)


class MissionArtifactCategoryWidget(QWidget):
    """
    Widget for categorizing mission artifacts.
    
    Wraps ArtifactCategoryWidget with mission-specific description.
    
    Signals:
        weights_changed(dict): Emitted when weights change
    """
    
    weights_changed = Signal(dict)
    
    def __init__(
        self,
        user_config: Optional[UserConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        
        config = user_config or UserConfig()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._category_widget = ArtifactCategoryWidget(
            title="Mission Artifact Categories",
            initial_weights=config.mission_artifact_weights,
        )
        self._category_widget.categories_changed.connect(self.weights_changed.emit)
        layout.addWidget(self._category_widget)
    
    def get_weights(self) -> Dict[str, float]:
        return self._category_widget.get_weights()
    
    def update_from_user_config(self, user_config: UserConfig) -> None:
        self._category_widget.update_from_weights(user_config.mission_artifact_weights)


class CraftedArtifactCategoryWidget(QWidget):
    """
    Widget for categorizing crafted artifact targets.
    
    Wraps ArtifactCategoryWidget with crafting-specific description.
    
    Signals:
        weights_changed(dict): Emitted when weights change
    """
    
    weights_changed = Signal(dict)
    
    def __init__(
        self,
        user_config: Optional[UserConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        
        config = user_config or UserConfig()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._category_widget = ArtifactCategoryWidget(
            title="Crafted Artifact Targets",
            initial_weights=config.crafted_artifact_weights,
        )
        self._category_widget.categories_changed.connect(self.weights_changed.emit)
        layout.addWidget(self._category_widget)
    
    def get_weights(self) -> Dict[str, float]:
        return self._category_widget.get_weights()
    
    def update_from_user_config(self, user_config: UserConfig) -> None:
        self._category_widget.update_from_weights(user_config.crafted_artifact_weights)
