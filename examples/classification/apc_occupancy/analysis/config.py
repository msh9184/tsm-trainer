"""Data paths, period definitions, sensor mapping, and device info.

All paths are designed for the 사내 GPU server by default.
Override ``DATA_ROOT`` when running from WSL with local sample data.

Period mapping
--------------
Four data collection periods with paired sensor + label files:

  P1: 2026-01-25 ~ 01-26 (2 days)
  P2: 2026-02-01 ~ 02-03 (3 days)
  P3: 2026-02-07 ~ 02-09 (3 days)
  P4: 2026-02-09 ~ 02-23 (15 days)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# Data root (override via CLI --data-root)
# ============================================================================
# Default: 사내 GPU server path
DEFAULT_DATA_ROOT = Path(
    "/group-volume/workspace/haeri.kim/Time-Series/data"
    "/SmartThings/Samsung_QST_Data/enter_leave"
)

# WSL local sample data path (for development / offline analysis)
WSL_SAMPLE_DATA_ROOT = Path("docs/sample_downstream_task")

# ============================================================================
# Period definitions
# ============================================================================

@dataclass
class PeriodDef:
    """Definition of a single data collection period."""
    name: str              # Human-readable name (e.g. "P1")
    sensor_file: str       # Filename of merged sensor CSV
    label_file: str        # Filename of occupancy events CSV
    date_range: str        # Human-readable date range
    initial_occupancy: int  # Inferred initial At-home count before first event


PERIODS: list[PeriodDef] = [
    PeriodDef(
        name="P1",
        sensor_file="merged_data_with_motion_count_0125_0126.csv",
        label_file="occupancy_events_0126_processed.csv",
        date_range="2026-01-25 ~ 01-26",
        initial_occupancy=2,
    ),
    PeriodDef(
        name="P2",
        sensor_file="merged_data_with_motion_count_0201_0203.csv",
        label_file="occupancy_events_0201_0203_processed.csv",
        date_range="2026-02-01 ~ 02-03",
        initial_occupancy=3,
    ),
    PeriodDef(
        name="P3",
        sensor_file="merged_data_with_motion_count_0207_0209.csv",
        label_file="occupancy_events_0207_0209_processed.csv",
        date_range="2026-02-07 ~ 02-09",
        initial_occupancy=3,
    ),
    PeriodDef(
        name="P4",
        sensor_file="merged_data_with_motion_count_0209_0223.csv",
        label_file="occupancy_events_0210_0219_processed.csv",
        date_range="2026-02-09 ~ 02-23",
        initial_occupancy=1,
    ),
]

# ============================================================================
# Sensor mapping: Device ID prefix -> physical device info
# ============================================================================

DEVICE_MAP: dict[str, dict[str, str]] = {
    "01976eca": {"name": "Aqara Presence Sensor", "location": "Room B",
                 "note": "Not accurate"},
    "ccea734e": {"name": "Aqara T&H Sensor T1", "location": "Room A",
                 "note": ""},
    "d620900d": {"name": "Motion Sensor", "location": "Entrance/Living room",
                 "note": "Also measures temperature, battery"},
    "408981c2": {"name": "Contact Sensor", "location": "Entrance door",
                 "note": "Door open/close"},
    "f2e891c6": {"name": "Smart Plug (Power)", "location": "Bathroom/Living room",
                 "note": "Dryer/Massager power monitoring"},
    "0fd9a9a4": {"name": "Smart Switch (Light)", "location": "Entrance light",
                 "note": ""},
    "7103039b": {"name": "Smart Lamp (Stand)", "location": "Living room",
                 "note": ""},
    "2ccd5a45": {"name": "Window Shade", "location": "Unknown",
                 "note": "Often near-constant or NaN"},
    "9a06ee71": {"name": "Window Shade 2", "location": "Unknown",
                 "note": ""},
}

# ============================================================================
# 15 common sensor columns (present in all 4 files)
# ============================================================================

COMMON_COLUMNS: list[str] = [
    "0fd9a9a4_switch",
    "2ccd5a45_windowShade",
    "408981c2_contactSensor",
    "7103039b_switch",
    "9a06ee71_windowShade",
    "9a06ee71_windowShadeLevel",
    "ccea734e_relativeHumidityMeasurement",
    "ccea734e_temperatureMeasurement",
    "d620900d_battery",
    "d620900d_motionSensor",
    "d620900d_temperatureMeasurement",
    "f2e891c6_energyMeter",
    "f2e891c6_powerConsumptionReport",
    "f2e891c6_powerMeter",
    "f2e891c6_switch",
]

# 8 core sensors recommended for classification
CORE_SENSORS: list[str] = [
    "d620900d_motionSensor",
    "408981c2_contactSensor",
    "d620900d_temperatureMeasurement",
    "ccea734e_temperatureMeasurement",
    "f2e891c6_switch",
    "f2e891c6_powerMeter",
    "0fd9a9a4_switch",
    "7103039b_switch",
]

# Sensor type categories for visualization grouping
SENSOR_CATEGORIES: dict[str, list[str]] = {
    "Motion / Presence": [
        "d620900d_motionSensor",
        "01976eca_motionSensor",
    ],
    "Contact (Door)": [
        "408981c2_contactSensor",
    ],
    "Temperature": [
        "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
    ],
    "Humidity": [
        "ccea734e_relativeHumidityMeasurement",
    ],
    "Power / Energy": [
        "f2e891c6_switch",
        "f2e891c6_powerMeter",
        "f2e891c6_energyMeter",
        "f2e891c6_powerConsumptionReport",
    ],
    "Light Switches": [
        "0fd9a9a4_switch",
        "7103039b_switch",
    ],
    "Window Shade": [
        "2ccd5a45_windowShade",
        "9a06ee71_windowShade",
        "9a06ee71_windowShadeLevel",
    ],
    "Battery": [
        "d620900d_battery",
    ],
}

# ============================================================================
# Visualization style palette (extends visualization/style.py)
# ============================================================================

# Period colors for Gantt charts and multi-period plots
PERIOD_COLORS: dict[str, str] = {
    "P1": "#4E79A7",  # Steel blue
    "P2": "#F28E2B",  # Orange
    "P3": "#E15759",  # Red-coral
    "P4": "#76B7B2",  # Teal
}

# Class colors (consistent with visualization/style.py)
CLASS_COLORS: dict[int, str] = {
    0: "#0173B2",  # Blue  — Empty
    1: "#DE8F05",  # Orange — Occupied
}
CLASS_NAMES: dict[int, str] = {
    0: "Empty",
    1: "Occupied",
}

# NaN heatmap colormap
NAN_CMAP = "YlOrRd"

# ============================================================================
# Analysis output config
# ============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for the full analysis pipeline."""
    data_root: Path = field(default_factory=lambda: DEFAULT_DATA_ROOT)
    output_dir: Path = field(default_factory=lambda: Path("results/apc_analysis"))
    dpi: int = 300
    formats: list[str] = field(default_factory=lambda: ["png"])
    figsize_wide: tuple[float, float] = (14, 5)
    figsize_square: tuple[float, float] = (8, 7)
    figsize_tall: tuple[float, float] = (10, 12)
    figsize_single: tuple[float, float] = (7, 5)
    bin_minutes: int = 5  # Sensor data binning interval
