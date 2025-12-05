"""
Main application window for the PySolve Eggs GUI.

Integrates all widgets and connects to the solver backend.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QGroupBox,
    QMessageBox,
    QStatusBar,
    QMenuBar,
    QMenu,
    QFileDialog,
    QApplication,
    QTabWidget,
    QDialog,
    QTextEdit,
    QDialogButtonBox,
    QInputDialog,
    QDockWidget,
)

from ..config import (
    UserConfig,
    EpicResearch,
    Constraints,
    CostWeights,
    load_config,
    save_config,
    DEFAULT_CONFIG_PATH,
    SHIP_METADATA,
)
from ..mission_solver import SolverResult, solve
from ..solver_logging import LogLevel, SolverLogger, create_string_logger
from ..solution_store import (
    SavedSolution,
    SolutionSummary,
    SolutionSource,
    MissionListItem,
    get_solution_store,
)
from ..bom import calculate_mission_list_score
from .widgets import (
    ShipConfigWidget,
    EpicResearchWidget,
    ConstraintsWidget,
    ResultsWidget,
    CombinedArtifactWidget,
    CostWeightsWidget,
    MissionListWidget,
    SolutionHistoryWidget,
    PlannerResultsWidget,
    ComparisonResultsWidget,
)


class SolverWorker(QObject):
    """
    Worker object to run the solver in a background thread.
    
    Signals:
        finished(SolverResult, str): Emitted when solver completes (result, log_text)
        error(str): Emitted on solver error
    """
    
    finished = Signal(object, str)  # SolverResult, log_text
    error = Signal(str)
    
    def __init__(self, config: UserConfig, num_ships: int, log_level: LogLevel = LogLevel.SILENT):
        super().__init__()
        self._config = config
        self._num_ships = num_ships
        self._log_level = log_level
    
    @Slot()
    def run(self) -> None:
        """Execute the solver."""
        try:
            logger, buffer = create_string_logger(level=self._log_level)
            result = solve(self._config, num_ships=self._num_ships, logger=logger)
            log_text = buffer.getvalue()
            self.finished.emit(result, log_text)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Layout:
    - Left panel: Ship configuration, Epic research, Constraints
    - Right panel: Results display
    - Bottom: Solve button, status bar
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._config = load_config(self._config_path)
        self._num_ships = 3
        self._solver_thread: Optional[QThread] = None
        self._dirty = False  # Track unsaved changes
        
        # Developer/logging state
        self._log_level = LogLevel.SILENT
        self._solver_logs: str = ""
        
        # Store last results for saving
        self._last_solver_result: Optional[SolverResult] = None
        self._last_planner_result: Optional[dict] = None
        
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        
        self.setWindowTitle("PySolve Eggs - Mission Optimizer")
        self.resize(1100, 750)
        self.setMinimumSize(900, 600)
    
    def _setup_ui(self) -> None:
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Main splitter: left config, right results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - configuration tabs (all config in tabs)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # Configuration tabs
        self._config_tabs = QTabWidget()
        
        # Tab 1: User Configuration (Ships + Epic Research)
        user_config_tab = QWidget()
        user_config_layout = QVBoxLayout(user_config_tab)
        user_config_layout.setContentsMargins(4, 4, 4, 4)
        user_config_layout.setSpacing(8)
        
        # Ship configuration
        ship_group = QGroupBox("Mission Configuration")
        ship_layout = QVBoxLayout(ship_group)
        self._ship_widget = ShipConfigWidget(self._config)
        ship_layout.addWidget(self._ship_widget)
        user_config_layout.addWidget(ship_group, stretch=3)
        
        # Epic research
        research_group = QGroupBox("Epic Research")
        research_layout = QVBoxLayout(research_group)
        self._research_widget = EpicResearchWidget(self._config)
        research_layout.addWidget(self._research_widget)
        user_config_layout.addWidget(research_group, stretch=1)
        
        self._config_tabs.addTab(user_config_tab, "User Config")
        
        # Tab 2: Combined Artifact Categories (replaces separate Collection/Crafting tabs)
        artifact_tab = QWidget()
        artifact_layout = QVBoxLayout(artifact_tab)
        artifact_layout.setContentsMargins(4, 4, 4, 4)
        self._artifact_widget = CombinedArtifactWidget()
        artifact_layout.addWidget(self._artifact_widget)
        self._config_tabs.addTab(artifact_tab, "Artifact Targets")
        
        # Tab 3: Solver Settings (Constraints + Priorities)
        solver_settings_tab = QWidget()
        solver_settings_layout = QVBoxLayout(solver_settings_tab)
        solver_settings_layout.setContentsMargins(4, 4, 4, 4)
        solver_settings_layout.setSpacing(12)
        
        # Constraints group
        constraints_group = QGroupBox("Constraints")
        constraints_group_layout = QVBoxLayout(constraints_group)
        self._constraints_widget = ConstraintsWidget(self._config, self._num_ships)
        constraints_group_layout.addWidget(self._constraints_widget)
        solver_settings_layout.addWidget(constraints_group)
        
        # Cost function weights (priorities)
        priorities_group = QGroupBox("Solver Priorities")
        priorities_layout = QVBoxLayout(priorities_group)
        self._cost_weights_widget = CostWeightsWidget(self._config)
        priorities_layout.addWidget(self._cost_weights_widget)
        solver_settings_layout.addWidget(priorities_group, 1)  # stretch factor 1 to fill space
        
        self._config_tabs.addTab(solver_settings_tab, "Solver Settings")
        
        # Tab 4: Mission Planner
        planner_tab = QWidget()
        planner_layout = QVBoxLayout(planner_tab)
        planner_layout.setContentsMargins(4, 4, 4, 4)
        planner_layout.setSpacing(8)
        
        self._mission_list_widget = MissionListWidget()
        planner_layout.addWidget(self._mission_list_widget)
        
        self._config_tabs.addTab(planner_tab, "Mission Planner")
        
        left_layout.addWidget(self._config_tabs)
        
        splitter.addWidget(left_panel)
        
        # Right panel - results (with tabs for different modes)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Results tabs: Optimizer | Planner | Comparison
        self._results_tabs = QTabWidget()
        
        # Optimizer results
        optimizer_results = QWidget()
        optimizer_layout = QVBoxLayout(optimizer_results)
        optimizer_layout.setContentsMargins(4, 4, 4, 4)
        self._results_widget = ResultsWidget()
        optimizer_layout.addWidget(self._results_widget)
        self._results_tabs.addTab(optimizer_results, "Optimizer")
        
        # Planner results
        planner_results = QWidget()
        planner_results_layout = QVBoxLayout(planner_results)
        planner_results_layout.setContentsMargins(4, 4, 4, 4)
        self._planner_results_widget = PlannerResultsWidget()
        planner_results_layout.addWidget(self._planner_results_widget)
        self._results_tabs.addTab(planner_results, "Planner")
        
        # Comparison results
        comparison_results = QWidget()
        comparison_layout = QVBoxLayout(comparison_results)
        comparison_layout.setContentsMargins(4, 4, 4, 4)
        self._comparison_widget = ComparisonResultsWidget()
        comparison_layout.addWidget(self._comparison_widget)
        self._results_tabs.addTab(comparison_results, "Comparison")
        
        results_group = QGroupBox("Results")
        results_group.setMinimumWidth(280)
        results_group_layout = QVBoxLayout(results_group)
        results_group_layout.setContentsMargins(4, 4, 4, 4)
        results_group_layout.addWidget(self._results_tabs)
        right_layout.addWidget(results_group)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions - balanced split
        splitter.setSizes([550, 450])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=1)
        
        # Bottom button row
        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        
        self._solve_btn = QPushButton("Solve")
        self._solve_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 10px 30px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self._solve_btn.clicked.connect(self._on_solve)
        button_row.addWidget(self._solve_btn)
        
        self._save_btn = QPushButton("Save Config")
        self._save_btn.clicked.connect(self._on_save_config)
        button_row.addWidget(self._save_btn)
        
        self._save_solution_btn = QPushButton("Save Solution")
        self._save_solution_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self._save_solution_btn.clicked.connect(self._on_save_solution)
        button_row.addWidget(self._save_solution_btn)
        
        button_row.addStretch()
        
        # Solution history button
        self._history_btn = QPushButton("📋 Solution History")
        self._history_btn.clicked.connect(self._toggle_history_dock)
        button_row.addWidget(self._history_btn)
        
        main_layout.addLayout(button_row)
        
        # Solution history dock widget
        self._history_dock = QDockWidget("Solution History", self)
        self._history_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        self._solution_history_widget = SolutionHistoryWidget()
        self._history_dock.setWidget(self._solution_history_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._history_dock)
        self._history_dock.hide()  # Start hidden
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")
    
    def _setup_menu(self) -> None:
        """Build the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        load_action = file_menu.addAction("&Load Config...")
        load_action.triggered.connect(self._on_load_config)
        
        save_action = file_menu.addAction("&Save Config")
        save_action.triggered.connect(self._on_save_config)
        
        save_as_action = file_menu.addAction("Save Config &As...")
        save_as_action.triggered.connect(self._on_save_config_as)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)
        
        # Developer menu
        dev_menu = menubar.addMenu("&Developer")
        
        # Log level submenu
        log_level_menu = dev_menu.addMenu("Log Level")
        self._log_level_group = QActionGroup(self)
        self._log_level_group.setExclusive(True)
        
        log_levels = [
            ("Silent", LogLevel.SILENT),
            ("Minimal", LogLevel.MINIMAL),
            ("Summary", LogLevel.SUMMARY),
            ("Detailed", LogLevel.DETAILED),
            ("Debug", LogLevel.DEBUG),
            ("Trace", LogLevel.TRACE),
        ]
        for name, level in log_levels:
            action = QAction(name, self)
            action.setCheckable(True)
            action.setChecked(level == self._log_level)
            action.setData(level)
            action.triggered.connect(self._on_log_level_changed)
            self._log_level_group.addAction(action)
            log_level_menu.addAction(action)
        
        dev_menu.addSeparator()
        
        view_logs_action = dev_menu.addAction("&View Logs...")
        view_logs_action.triggered.connect(self._on_view_logs)
        
        export_logs_action = dev_menu.addAction("&Export Logs...")
        export_logs_action.triggered.connect(self._on_export_logs)
        
        clear_logs_action = dev_menu.addAction("&Clear Logs")
        clear_logs_action.triggered.connect(self._on_clear_logs)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self._on_about)
    
    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._ship_widget.config_changed.connect(self._on_config_changed)
        self._research_widget.config_changed.connect(self._on_config_changed)
        self._constraints_widget.config_changed.connect(self._on_constraints_changed)
        self._cost_weights_widget.weights_changed.connect(self._on_config_changed)
        self._artifact_widget.weights_changed.connect(self._on_config_changed)
        
        # Mission Planner signals
        self._mission_list_widget.calculate_requested.connect(self._on_calculate_planner)
        
        # Solution history signals
        self._solution_history_widget.solution_load_requested.connect(self._on_load_solution)
        self._solution_history_widget.solutions_compare_requested.connect(self._on_compare_solutions)
    
    def _on_config_changed(self, *args) -> None:
        """Handle any configuration change."""
        self._dirty = True
        self._update_window_title()
    
    def _on_constraints_changed(self, constraints: Constraints, num_ships: int) -> None:
        """Handle constraints change."""
        self._num_ships = num_ships
        self._dirty = True
        self._update_window_title()
    
    def _update_window_title(self) -> None:
        """Update window title to show dirty state."""
        title = "PySolve Eggs - Mission Optimizer"
        if self._dirty:
            title += " *"
        self.setWindowTitle(title)
    
    def _build_config_from_widgets(self) -> UserConfig:
        """Build a UserConfig from current widget states."""
        # Get ship missions (excluding ships with level -1)
        missions_dict = self._ship_widget.get_missions_dict()
        missions = {
            ship: level for ship, level in missions_dict.items()
            if level >= 0
        }
        
        # Epic research
        epic_researches = self._research_widget.get_epic_researches()
        
        # Constraints
        constraints = self._constraints_widget.get_constraints()
        
        # Cost weights from the new widget
        cost_weights = self._cost_weights_widget.get_cost_weights()
        
        # Artifact weights from the combined widget
        # Returns (ship_weights, craft_weights) tuple
        ship_weights, craft_weights = self._artifact_widget.get_weights()
        
        return UserConfig(
            missions=missions,
            epic_researches=epic_researches,
            constraints=constraints,
            cost_weights=cost_weights,
            crafted_artifact_weights=craft_weights,
            mission_artifact_weights=ship_weights,
        )
    
    @Slot()
    def _on_solve(self) -> None:
        """Run the solver."""
        if self._solver_thread is not None and self._solver_thread.isRunning():
            return  # Already running
        
        # Build config from widgets
        config = self._build_config_from_widgets()
        
        # Show running state
        self._solve_btn.setEnabled(False)
        self._results_widget.set_running()
        self._status_bar.showMessage("Solving...")
        
        # Create worker and thread
        self._solver_thread = QThread()
        self._solver_worker = SolverWorker(config, self._num_ships, self._log_level)
        self._solver_worker.moveToThread(self._solver_thread)
        
        # Connect signals
        self._solver_thread.started.connect(self._solver_worker.run)
        self._solver_worker.finished.connect(self._on_solve_finished)
        self._solver_worker.error.connect(self._on_solve_error)
        self._solver_worker.finished.connect(self._solver_thread.quit)
        self._solver_worker.error.connect(self._solver_thread.quit)
        
        # Start
        self._solver_thread.start()
    
    @Slot(object, str)
    def _on_solve_finished(self, result: SolverResult, log_text: str) -> None:
        """Handle solver completion."""
        self._solve_btn.setEnabled(True)
        
        # Append logs
        if log_text:
            if self._solver_logs:
                self._solver_logs += "\n" + "=" * 60 + "\n"
            self._solver_logs += log_text
        
        fuel_capacity = self._constraints_widget.get_constraints().fuel_tank_capacity
        self._results_widget.set_result(result, fuel_capacity)
        
        # Store result for saving
        self._last_solver_result = result
        
        # Switch to optimizer results tab
        self._results_tabs.setCurrentIndex(0)
        
        self._status_bar.showMessage(
            f"Solved: {result.status}, {len(result.selected_missions)} missions, "
            f"{result.total_time_hours:.1f} hours"
        )
    
    @Slot(str)
    def _on_solve_error(self, error_msg: str) -> None:
        """Handle solver error."""
        self._solve_btn.setEnabled(True)
        self._results_widget.clear()
        
        self._status_bar.showMessage(f"Error: {error_msg}")
        QMessageBox.critical(self, "Solver Error", f"Failed to solve:\n{error_msg}")
    
    @Slot()
    def _on_save_config(self) -> None:
        """Save configuration to current path."""
        config = self._build_config_from_widgets()
        try:
            save_config(config, self._config_path)
            self._config = config
            self._dirty = False
            self._update_window_title()
            self._status_bar.showMessage(f"Saved to {self._config_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
    
    @Slot()
    def _on_save_config_as(self) -> None:
        """Save configuration to a new path."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            str(self._config_path.parent),
            "YAML Files (*.yaml);;All Files (*)",
        )
        if path:
            self._config_path = Path(path)
            self._on_save_config()
    
    @Slot()
    def _on_load_config(self) -> None:
        """Load configuration from file."""
        if self._dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Load anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            str(self._config_path.parent),
            "YAML Files (*.yaml);;All Files (*)",
        )
        if path:
            try:
                self._config_path = Path(path)
                self._config = load_config(self._config_path)
                
                # Update widgets
                self._ship_widget.update_from_user_config(self._config)
                self._research_widget.update_from_user_config(self._config)
                self._constraints_widget.update_from_user_config(self._config, self._num_ships)
                self._cost_weights_widget.update_from_user_config(self._config)
                self._artifact_widget.update_from_user_config(self._config)
                
                self._dirty = False
                self._update_window_title()
                self._status_bar.showMessage(f"Loaded from {self._config_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load:\n{e}")
    
    @Slot()
    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PySolve Eggs",
            "<h2>PySolve Eggs</h2>"
            "<p>Mission optimizer for Egg Inc. using linear programming.</p>"
            "<p>Version 0.1.0</p>"
            "<p>Built with PySide6 and PuLP</p>",
        )
    
    def closeEvent(self, event) -> None:
        """Handle window close."""
        if self._dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_config()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    # -------------------------------------------------------------------------
    # Developer Menu Handlers
    # -------------------------------------------------------------------------
    
    @Slot()
    def _on_log_level_changed(self) -> None:
        """Handle log level radio button change."""
        action = self._log_level_group.checkedAction()
        if action:
            self._log_level = action.data()
            level_name = self._log_level.name.title()
            self._status_bar.showMessage(f"Log level set to {level_name}", 3000)
    
    @Slot()
    def _on_view_logs(self) -> None:
        """Show a dialog with the solver logs."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Solver Logs")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Consolas, Monaco, monospace")
        if self._solver_logs:
            text_edit.setPlainText(self._solver_logs)
        else:
            text_edit.setPlainText("(No logs yet. Run the solver with a log level other than Silent.)")
        layout.addWidget(text_edit)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.exec()
    
    @Slot()
    def _on_export_logs(self) -> None:
        """Export solver logs to a file."""
        if not self._solver_logs:
            QMessageBox.information(
                self,
                "No Logs",
                "No logs to export. Run the solver with a log level other than Silent.",
            )
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Logs",
            "solver_logs.txt",
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)",
        )
        if path:
            try:
                Path(path).write_text(self._solver_logs, encoding="utf-8")
                self._status_bar.showMessage(f"Logs exported to {path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export logs:\n{e}")
    
    @Slot()
    def _on_clear_logs(self) -> None:
        """Clear all stored solver logs."""
        self._solver_logs = ""
        self._status_bar.showMessage("Logs cleared", 3000)

    # -------------------------------------------------------------------------
    # Mission Planner Handlers
    # -------------------------------------------------------------------------
    
    @Slot()
    def _on_calculate_planner(self) -> None:
        """Calculate results for the mission planner list."""
        missions = self._mission_list_widget.get_mission_tuples()
        
        if not missions:
            QMessageBox.information(
                self,
                "No Missions",
                "Please add at least one mission to the list.",
            )
            return
        
        # Get crafting/mission weights from artifact widget
        ship_weights, craft_weights = self._artifact_widget.get_weights()
        
        # Get capacity bonus from epic research
        epic_researches = self._research_widget.get_epic_researches()
        capacity_bonus = 0.0
        zgqc = epic_researches.get("Zero-G Quantum Containment")
        if zgqc:
            capacity_bonus = zgqc.level * zgqc.effect
        
        # Get mission level (use first ship's level or default)
        missions_dict = self._ship_widget.get_missions_dict()
        first_ship = missions[0][0] if missions else "HENERPRISE"
        mission_level = missions_dict.get(first_ship, 0)
        
        try:
            score, total_drops, bom_rollup = calculate_mission_list_score(
                missions=missions,
                crafting_weights=craft_weights if craft_weights else None,
                mission_weights=ship_weights if ship_weights else None,
                mission_level=mission_level,
                capacity_bonus=capacity_bonus,
            )
            
            # Update planner results
            self._planner_results_widget.set_result(score, total_drops, bom_rollup)
            
            # Switch to planner results tab
            self._results_tabs.setCurrentIndex(1)
            
            # Store last calculation for saving
            self._last_planner_result = {
                "score": score,
                "total_drops": total_drops,
                "bom_rollup": bom_rollup,
                "missions": self._mission_list_widget.get_mission_list(),
            }
            
            self._status_bar.showMessage(
                f"Calculated: Score {score:.2f}, {sum(total_drops.values()):.1f} total drops"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate:\n{e}")
    
    @Slot()
    def _on_save_solution(self) -> None:
        """Save the current solution (optimizer or planner)."""
        store = get_solution_store()
        
        # Determine which tab is active
        current_tab = self._results_tabs.currentIndex()
        
        if current_tab == 0:  # Optimizer
            source_type = SolutionSource.OPTIMIZER
            
            # Check if we have optimizer results
            if not hasattr(self, '_last_solver_result') or self._last_solver_result is None:
                QMessageBox.information(
                    self,
                    "No Solution",
                    "No optimizer solution to save. Run the solver first.",
                )
                return
            
            result = self._last_solver_result
            mission_list = [
                MissionListItem(
                    ship=m.ship,
                    ship_label=m.ship_label,
                    duration=m.duration_type,
                    level=m.level,
                    target=m.target_artifact,
                    count=count,
                )
                for m, count in result.selected_missions
            ]
            
            summary = SolutionSummary(
                status=result.status,
                objective_value=result.objective_value,
                total_time_hours=result.total_time_hours,
                total_drops=result.total_drops,
                crafted=result.bom_rollup.crafted if result.bom_rollup else {},
                consumed=result.bom_rollup.consumed if result.bom_rollup else {},
                remaining=result.bom_rollup.remaining if result.bom_rollup else {},
            )
            
        elif current_tab == 1:  # Planner
            source_type = SolutionSource.MISSION_LIST
            
            # Check if we have planner results
            if not hasattr(self, '_last_planner_result') or self._last_planner_result is None:
                QMessageBox.information(
                    self,
                    "No Calculation",
                    "No planner calculation to save. Click Calculate first.",
                )
                return
            
            planner = self._last_planner_result
            mission_list = planner["missions"]
            
            summary = SolutionSummary(
                status="Calculated",
                objective_value=planner["score"],
                total_time_hours=0.0,  # TODO: Calculate time
                total_drops=planner["total_drops"],
                crafted=planner["bom_rollup"].crafted if planner["bom_rollup"] else {},
                consumed=planner["bom_rollup"].consumed if planner["bom_rollup"] else {},
                remaining=planner["bom_rollup"].remaining if planner["bom_rollup"] else {},
            )
            
        else:  # Comparison tab - nothing to save
            QMessageBox.information(
                self,
                "Cannot Save",
                "Cannot save from comparison view. Switch to Optimizer or Planner results.",
            )
            return
        
        # Generate name
        name = store.generate_name(source_type)
        timestamp = datetime.now().isoformat()
        
        # Ask for optional display name
        display_name, ok = QInputDialog.getText(
            self,
            "Save Solution",
            "Display name (or leave empty for auto-name):",
            text=name,
        )
        
        if not ok:
            return  # Cancelled
        
        if not display_name.strip():
            display_name = name
        
        # Create and save solution
        solution = SavedSolution(
            name=name,
            display_name=display_name.strip(),
            timestamp=timestamp,
            source_type=source_type,
            mission_list=mission_list,
            result=summary,
        )
        
        try:
            path = store.save_solution(solution)
            self._status_bar.showMessage(f"Solution saved: {display_name}", 5000)
            
            # Refresh history if visible
            if self._history_dock.isVisible():
                self._solution_history_widget.refresh()
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save solution:\n{e}")
    
    @Slot()
    def _toggle_history_dock(self) -> None:
        """Toggle the solution history dock visibility."""
        if self._history_dock.isVisible():
            self._history_dock.hide()
        else:
            self._history_dock.show()
            self._solution_history_widget.refresh()
    
    @Slot(str)
    def _on_load_solution(self, name: str) -> None:
        """Load a solution into the mission planner."""
        store = get_solution_store()
        solution = store.load_solution(name)
        
        if solution is None:
            QMessageBox.warning(self, "Load Error", f"Could not load solution: {name}")
            return
        
        # Load mission list into planner
        self._mission_list_widget.set_mission_list(solution.mission_list)
        
        # Switch to Mission Planner tab
        self._config_tabs.setCurrentIndex(3)  # Mission Planner tab
        
        self._status_bar.showMessage(f"Loaded solution: {solution.display_name}", 5000)
    
    @Slot(list)
    def _on_compare_solutions(self, names: List[str]) -> None:
        """Compare selected solutions."""
        store = get_solution_store()
        solutions = store.get_solutions_by_names(names)
        
        if len(solutions) < 2:
            QMessageBox.warning(
                self,
                "Comparison Error",
                "Need at least 2 solutions to compare.",
            )
            return
        
        self._comparison_widget.set_solutions(solutions)
        
        # Switch to comparison results tab
        self._results_tabs.setCurrentIndex(2)
        
        self._status_bar.showMessage(f"Comparing {len(solutions)} solutions", 5000)
