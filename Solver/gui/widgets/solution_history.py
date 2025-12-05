"""
Solution history browser widget.

Displays saved solutions with multi-select for comparison,
inline renaming, and load/delete actions.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QLabel,
    QMessageBox,
    QAbstractItemView,
    QLineEdit,
    QStyledItemDelegate,
)

from ...solution_store import SavedSolution, SolutionStore, get_solution_store


# Maximum number of solutions that can be compared
MAX_COMPARISON_SOLUTIONS = 4


class EditableDelegate(QStyledItemDelegate):
    """Delegate that allows editing only the display name column."""
    
    def __init__(self, editable_column: int, parent=None):
        super().__init__(parent)
        self._editable_column = editable_column
    
    def createEditor(self, parent, option, index):
        if index.column() == self._editable_column:
            editor = QLineEdit(parent)
            return editor
        return None


class SolutionHistoryWidget(QWidget):
    """
    Widget for browsing and managing saved solutions.
    
    Features:
    - Table showing saved solutions (Name, Display Name, Date, Type, Score)
    - Multi-select for comparison (max 4)
    - Inline editing of display names (double-click)
    - Load, Delete, Compare actions
    
    Signals
    -------
    solution_load_requested : Signal(str)
        Emitted when user wants to load a solution (passes solution name)
    solutions_compare_requested : Signal(list)
        Emitted when user wants to compare solutions (passes list of names)
    """
    
    solution_load_requested = Signal(str)
    solutions_compare_requested = Signal(list)
    
    # Column indices
    COL_NAME = 0
    COL_DISPLAY_NAME = 1
    COL_DATE = 2
    COL_TYPE = 3
    COL_SCORE = 4
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._store = get_solution_store()
        self._solutions: List[SavedSolution] = []
        
        self._setup_ui()
        self._connect_signals()
        self.refresh()
    
    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Saved Solutions")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self._refresh_btn = QPushButton("↻ Refresh")
        self._refresh_btn.setFixedWidth(80)
        header_layout.addWidget(self._refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Name", "Display Name", "Date", "Type", "Score"
        ])
        
        # Configure columns
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_DISPLAY_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_DATE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_TYPE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_SCORE, QHeaderView.ResizeMode.ResizeToContents)
        
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        # Enable inline editing for display name column only
        delegate = EditableDelegate(self.COL_DISPLAY_NAME, self._table)
        self._table.setItemDelegate(delegate)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed
        )
        
        layout.addWidget(self._table, stretch=1)
        
        # Selection info
        self._selection_label = QLabel("Select solutions to compare (max 4)")
        self._selection_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self._selection_label)
        
        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        self._load_btn = QPushButton("Load")
        self._load_btn.setEnabled(False)
        btn_row.addWidget(self._load_btn)
        
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        self._delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        btn_row.addWidget(self._delete_btn)
        
        btn_row.addStretch()
        
        self._compare_btn = QPushButton("Compare Selected")
        self._compare_btn.setEnabled(False)
        self._compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        btn_row.addWidget(self._compare_btn)
        
        layout.addLayout(btn_row)
    
    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._refresh_btn.clicked.connect(self.refresh)
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        self._load_btn.clicked.connect(self._on_load)
        self._delete_btn.clicked.connect(self._on_delete)
        self._compare_btn.clicked.connect(self._on_compare)
        self._table.doubleClicked.connect(self._on_double_click)
    
    def refresh(self) -> None:
        """Reload solutions from disk."""
        self._solutions = self._store.list_solutions()
        self._populate_table()
    
    def _populate_table(self) -> None:
        """Populate the table with solutions."""
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        
        for solution in self._solutions:
            row = self._table.rowCount()
            self._table.insertRow(row)
            
            # Name (not editable)
            name_item = QTableWidgetItem(solution.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name_item.setData(Qt.ItemDataRole.UserRole, solution.name)
            self._table.setItem(row, self.COL_NAME, name_item)
            
            # Display Name (editable via double-click)
            display_item = QTableWidgetItem(solution.display_name)
            self._table.setItem(row, self.COL_DISPLAY_NAME, display_item)
            
            # Date
            date_str = solution.timestamp[:10] if solution.timestamp else ""
            date_item = QTableWidgetItem(date_str)
            date_item.setFlags(date_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, self.COL_DATE, date_item)
            
            # Type
            type_item = QTableWidgetItem(solution.source_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            # Color-code by type
            if solution.source_type == "Solution":
                type_item.setForeground(Qt.GlobalColor.darkBlue)
            else:
                type_item.setForeground(Qt.GlobalColor.darkGreen)
            self._table.setItem(row, self.COL_TYPE, type_item)
            
            # Score
            score_str = f"{solution.result.objective_value:.2f}"
            score_item = QTableWidgetItem(score_str)
            score_item.setFlags(score_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, self.COL_SCORE, score_item)
        
        self._table.blockSignals(False)
        self._on_selection_changed()
    
    def _on_selection_changed(self) -> None:
        """Handle selection changes."""
        selected = self._get_selected_names()
        count = len(selected)
        
        self._load_btn.setEnabled(count == 1)
        self._delete_btn.setEnabled(count >= 1)
        self._compare_btn.setEnabled(2 <= count <= MAX_COMPARISON_SOLUTIONS)
        
        if count > MAX_COMPARISON_SOLUTIONS:
            self._selection_label.setText(
                f"⚠ Too many selected ({count}). Max {MAX_COMPARISON_SOLUTIONS} for comparison."
            )
            self._selection_label.setStyleSheet("color: #e74c3c; font-style: italic;")
        elif count >= 2:
            self._selection_label.setText(f"✓ {count} solutions selected for comparison")
            self._selection_label.setStyleSheet("color: #27ae60; font-style: italic;")
        elif count == 1:
            self._selection_label.setText("Select 1 more solution to compare")
            self._selection_label.setStyleSheet("color: #666; font-style: italic;")
        else:
            self._selection_label.setText(f"Select solutions to compare (max {MAX_COMPARISON_SOLUTIONS})")
            self._selection_label.setStyleSheet("color: #666; font-style: italic;")
    
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Handle item edits (display name rename)."""
        if item.column() != self.COL_DISPLAY_NAME:
            return
        
        row = item.row()
        name_item = self._table.item(row, self.COL_NAME)
        if not name_item:
            return
        
        name = name_item.data(Qt.ItemDataRole.UserRole)
        new_display_name = item.text().strip()
        
        if new_display_name:
            self._store.rename_solution(name, new_display_name)
    
    def _on_double_click(self, index) -> None:
        """Handle double-click (load if not on editable column)."""
        if index.column() != self.COL_DISPLAY_NAME:
            self._on_load()
    
    def _on_load(self) -> None:
        """Load the selected solution."""
        selected = self._get_selected_names()
        if len(selected) == 1:
            self.solution_load_requested.emit(selected[0])
    
    def _on_delete(self) -> None:
        """Delete selected solutions."""
        selected = self._get_selected_names()
        if not selected:
            return
        
        count = len(selected)
        msg = f"Delete {count} solution(s)?" if count > 1 else "Delete this solution?"
        
        reply = QMessageBox.question(
            self,
            "Delete Solutions",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for name in selected:
                self._store.delete_solution(name)
            self.refresh()
    
    def _on_compare(self) -> None:
        """Compare selected solutions."""
        selected = self._get_selected_names()
        if 2 <= len(selected) <= MAX_COMPARISON_SOLUTIONS:
            self.solutions_compare_requested.emit(selected)
    
    def _get_selected_names(self) -> List[str]:
        """Get names of selected solutions."""
        names = []
        for row in set(index.row() for index in self._table.selectedIndexes()):
            name_item = self._table.item(row, self.COL_NAME)
            if name_item:
                names.append(name_item.data(Qt.ItemDataRole.UserRole))
        return names
    
    def get_selected_solutions(self) -> List[SavedSolution]:
        """Get the full SavedSolution objects for selected rows."""
        names = self._get_selected_names()
        return self._store.get_solutions_by_names(names)
