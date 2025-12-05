"""
Cost function weights widget.

Provides controls for solver priority weights:
- Efficiency factors (fuel, time, waste) with scale and power parameters
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
    QFrame,
    QScrollArea,
)

from ...config import CostWeights, UserConfig


class WeightSliderWidget(QWidget):
    """
    A labeled slider with numeric display for weight configuration.
    
    Signals:
        value_changed(float): Emitted when value changes
    """
    
    value_changed = Signal(float)
    
    def __init__(
        self,
        label: str,
        description: str,
        min_val: float = 0.0,
        max_val: float = 10.0,
        initial: float = 1.0,
        decimals: int = 1,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._scale = 10 ** decimals
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        
        # Label row
        label_row = QHBoxLayout()
        
        name_label = QLabel(label)
        name_label.setStyleSheet("font-weight: bold;")
        label_row.addWidget(name_label)
        
        label_row.addStretch()
        
        # Value display
        self._value_spin = QDoubleSpinBox()
        self._value_spin.setRange(min_val, max_val)
        self._value_spin.setDecimals(decimals)
        self._value_spin.setValue(initial)
        self._value_spin.setFixedWidth(90)
        self._value_spin.valueChanged.connect(self._on_spin_changed)
        label_row.addWidget(self._value_spin)
        
        layout.addLayout(label_row)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #666; font-size: 11px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val * self._scale), int(max_val * self._scale))
        self._slider.setValue(int(initial * self._scale))
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)
    
    @property
    def value(self) -> float:
        return self._value_spin.value()
    
    @value.setter
    def value(self, val: float) -> None:
        self._value_spin.blockSignals(True)
        self._slider.blockSignals(True)
        
        self._value_spin.setValue(val)
        self._slider.setValue(int(val * self._scale))
        
        self._value_spin.blockSignals(False)
        self._slider.blockSignals(False)
    
    def _on_slider_changed(self, value: int) -> None:
        float_val = value / self._scale
        self._value_spin.blockSignals(True)
        self._value_spin.setValue(float_val)
        self._value_spin.blockSignals(False)
        self.value_changed.emit(float_val)
    
    def _on_spin_changed(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(value * self._scale))
        self._slider.blockSignals(False)
        self.value_changed.emit(value)


class PowerSliderWidget(QWidget):
    """
    A slider with tick marks at 1, 2, 3 labeled Linear, Quadratic, Cubic.
    
    Signals:
        value_changed(float): Emitted when value changes
    """
    
    value_changed = Signal(float)
    
    def __init__(
        self,
        label: str,
        description: str,
        min_val: float = 0.0,
        max_val: float = 3.0,
        initial: float = 0.0,
        decimals: int = 1,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._scale = 10 ** decimals
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        
        # Label row
        label_row = QHBoxLayout()
        
        name_label = QLabel(label)
        name_label.setStyleSheet("font-weight: bold;")
        label_row.addWidget(name_label)
        
        label_row.addStretch()
        
        # Value display
        self._value_spin = QDoubleSpinBox()
        self._value_spin.setRange(min_val, max_val)
        self._value_spin.setDecimals(decimals)
        self._value_spin.setValue(initial)
        self._value_spin.setFixedWidth(90)
        self._value_spin.valueChanged.connect(self._on_spin_changed)
        label_row.addWidget(self._value_spin)
        
        layout.addLayout(label_row)
        
        # Description with tick label info integrated
        desc_label = QLabel(f"{description} (1=Linear, 2=Quadratic, 3=Cubic)")
        desc_label.setStyleSheet("color: #666; font-size: 11px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Slider with tick marks
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val * self._scale), int(max_val * self._scale))
        self._slider.setValue(int(initial * self._scale))
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(int(1.0 * self._scale))  # Tick every 1.0
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)
    
    @property
    def value(self) -> float:
        return self._value_spin.value()
    
    @value.setter
    def value(self, val: float) -> None:
        self._value_spin.blockSignals(True)
        self._slider.blockSignals(True)
        
        self._value_spin.setValue(val)
        self._slider.setValue(int(val * self._scale))
        
        self._value_spin.blockSignals(False)
        self._slider.blockSignals(False)
    
    def _on_slider_changed(self, value: int) -> None:
        float_val = value / self._scale
        self._value_spin.blockSignals(True)
        self._value_spin.setValue(float_val)
        self._value_spin.blockSignals(False)
        self.value_changed.emit(float_val)
    
    def _on_spin_changed(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(value * self._scale))
        self._slider.blockSignals(False)
        self.value_changed.emit(value)


class CostWeightsWidget(QWidget):
    """
    Widget for configuring cost function weights.
    
    Controls:
    - Efficiency Factors (fuel, time, waste) with scale and power parameters
    
    Signals:
        weights_changed(CostWeights): Emitted when any weight changes
    """
    
    weights_changed = Signal(object)  # CostWeights
    
    def __init__(
        self,
        user_config: Optional[UserConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        
        config = user_config or UserConfig()
        weights = config.cost_weights
        
        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Scroll area for efficiency groups
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Description of efficiency factors
        desc_label = QLabel(
            "Efficiency factors adjust how the solver values missions. "
            "When enabled (power > 0), they multiply each mission's score to favor "
            "fuel-efficient, time-efficient, or waste-free missions over raw artifact counts."
        )
        desc_label.setStyleSheet("color: #666; font-size: 11px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Fuel Efficiency
        fuel_group = QGroupBox("Fuel Efficiency")
        fuel_group.setMinimumHeight(140)  # Prevent shrinking
        fuel_group_layout = QVBoxLayout(fuel_group)
        
        fuel_desc = QLabel("Penalizes missions with higher fuel cost per artifact")
        fuel_desc.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        fuel_desc.setWordWrap(True)
        fuel_group_layout.addWidget(fuel_desc)
        
        fuel_sliders = QHBoxLayout()
        
        self._fuel_scale = WeightSliderWidget(
            label="Scale",
            description="Base multiplier for fuel ratio (≥1.0)",
            min_val=1.0,
            max_val=100.0,
            initial=weights.fuel_efficiency_scale,
        )
        self._fuel_scale.value_changed.connect(self._on_changed)
        fuel_sliders.addWidget(self._fuel_scale)
        
        self._fuel_power = PowerSliderWidget(
            label="Power",
            description="Exponent (0=ignore, higher=stronger effect)",
            min_val=0.0,
            max_val=3.0,
            initial=weights.fuel_efficiency_power,
        )
        self._fuel_power.value_changed.connect(self._on_changed)
        fuel_sliders.addWidget(self._fuel_power)
        
        fuel_group_layout.addLayout(fuel_sliders)
        layout.addWidget(fuel_group)
        
        # Time Efficiency
        time_group = QGroupBox("Time Efficiency")
        time_group.setMinimumHeight(140)  # Prevent shrinking
        time_group_layout = QVBoxLayout(time_group)
        
        time_desc = QLabel("Rewards missions with more artifacts per hour")
        time_desc.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        time_desc.setWordWrap(True)
        time_group_layout.addWidget(time_desc)
        
        time_sliders = QHBoxLayout()
        
        self._time_scale = WeightSliderWidget(
            label="Scale",
            description="Base multiplier for time ratio (≥1.0)",
            min_val=1.0,
            max_val=100.0,
            initial=weights.time_efficiency_scale,
        )
        self._time_scale.value_changed.connect(self._on_changed)
        time_sliders.addWidget(self._time_scale)
        
        self._time_power = PowerSliderWidget(
            label="Power",
            description="Exponent (0=ignore, higher=stronger effect)",
            min_val=0.0,
            max_val=3.0,
            initial=weights.time_efficiency_power,
        )
        self._time_power.value_changed.connect(self._on_changed)
        time_sliders.addWidget(self._time_power)
        
        time_group_layout.addLayout(time_sliders)
        layout.addWidget(time_group)
        
        # Waste Efficiency
        waste_group = QGroupBox("Waste Efficiency")
        waste_group.setMinimumHeight(140)  # Prevent shrinking
        waste_group_layout = QVBoxLayout(waste_group)
        
        waste_desc = QLabel("Penalizes missions that drop non-targeted artifacts")
        waste_desc.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        waste_desc.setWordWrap(True)
        waste_group_layout.addWidget(waste_desc)
        
        waste_sliders = QHBoxLayout()
        
        self._waste_scale = WeightSliderWidget(
            label="Scale",
            description="Base multiplier for waste ratio (≥1.0)",
            min_val=1.0,
            max_val=100.0,
            initial=weights.waste_efficiency_scale,
        )
        self._waste_scale.value_changed.connect(self._on_changed)
        waste_sliders.addWidget(self._waste_scale)
        
        self._waste_power = PowerSliderWidget(
            label="Power",
            description="Exponent (0=ignore, higher=stronger effect)",
            min_val=0.0,
            max_val=3.0,
            initial=weights.waste_efficiency_power,
        )
        self._waste_power.value_changed.connect(self._on_changed)
        waste_sliders.addWidget(self._waste_power)
        
        waste_group_layout.addLayout(waste_sliders)
        layout.addWidget(waste_group)
        
        # Set scroll area content and add to main layout
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # Set minimum height for widget to fit at least one group
        self.setMinimumHeight(200)
    
    def _on_changed(self, *args) -> None:
        """Handle any weight change."""
        self.weights_changed.emit(self.get_cost_weights())
    
    def get_cost_weights(self) -> CostWeights:
        """Get current CostWeights object."""
        return CostWeights(
            fuel_efficiency_scale=self._fuel_scale.value,
            fuel_efficiency_power=self._fuel_power.value,
            time_efficiency_scale=self._time_scale.value,
            time_efficiency_power=self._time_power.value,
            waste_efficiency_scale=self._waste_scale.value,
            waste_efficiency_power=self._waste_power.value,
        )
    
    def update_from_user_config(self, user_config: UserConfig) -> None:
        """Update from UserConfig."""
        weights = user_config.cost_weights
        self._fuel_scale.value = weights.fuel_efficiency_scale
        self._fuel_power.value = weights.fuel_efficiency_power
        self._time_scale.value = weights.time_efficiency_scale
        self._time_power.value = weights.time_efficiency_power
        self._waste_scale.value = weights.waste_efficiency_scale
        self._waste_power.value = weights.waste_efficiency_power
