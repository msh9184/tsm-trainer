# APC Occupancy Data Analysis & EDA Toolkit

Comprehensive exploratory data analysis (EDA) toolkit for SmartThings sensor data
used in binary occupancy detection. Generates publication-quality visualizations
covering 7 analysis dimensions with 27 total plots.

## Quick Start

```bash
# From 사내 GPU server
cd /group-volume/workspace/sunghwan.mun/mygit3/tsm-trainer
python -m examples.classification.apc_occupancy.analysis.run_all \
    --data-root /group-volume/workspace/haeri.kim/Time-Series/data/SmartThings/Samsung_QST_Data/enter_leave/ \
    --output-dir results/apc_analysis/

# From WSL (local sample data for development)
cd /mnt/c/Users/User/workspace/samsung_research/sunghwan.mun/mygit3/tsm-trainer
python -m examples.classification.apc_occupancy.analysis.run_all \
    --data-root docs/sample_downstream_task/ \
    --output-dir results/apc_analysis/
```

## Analysis Modules

| Module | Description | Plots |
|--------|-------------|-------|
| **01_overview** | Summary statistics table, data volume bars, column inventory heatmap | 3 |
| **02_class_balance** | Per-period class distribution, overall balance donut, hourly occupancy heatmap, occupancy timeline | 4 |
| **03_nan_analysis** | NaN fraction heatmap (sensor × period), NaN bars for common columns, temporal NaN density | 3 |
| **04_time_coverage** | Gantt chart (sensor vs label ranges), inter-period gap timeline, overlap percentage matrix | 3 |
| **05_sensor_timeline** | Core sensor dashboard with occupancy overlay (×4 periods), all-sensors normalized view (×4) | 8 |
| **06_correlation** | Sensor-sensor Pearson correlation matrix, point-biserial sensor-label correlation, class-conditional boxplots | 3 |
| **07_gap_analysis** | Time interval histogram, gap detection summary, event timing analysis (inter-event, type breakdown, hourly) | 3 |
| **Total** | | **27** |

## CLI Options

```
--data-root PATH      Root directory containing CSV files (required)
--output-dir PATH     Output directory for plots (default: results/apc_analysis/)
--dpi INT             Output resolution (default: 300)
--formats STR         Output format(s), comma-separated (default: png)
--modules STR         Modules to run, comma-separated (default: all)
                      Options: 01_overview, 02_class_balance, 03_nan,
                               04_coverage, 05_sensor, 06_correlation, 07_gap
--verbose / -v        Enable verbose logging
```

### Run specific modules only

```bash
python -m examples.classification.apc_occupancy.analysis.run_all \
    --data-root docs/sample_downstream_task/ \
    --output-dir results/apc_analysis/ \
    --modules 02_class_balance,06_correlation
```

## Output Directory Structure

```
results/apc_analysis/
├── 01_overview/
│   ├── summary_table.png           # Dataset summary statistics table
│   ├── data_volume.png             # Timestep count + event count bars
│   └── column_inventory.png        # Sensor column presence heatmap
├── 02_class_balance/
│   ├── class_distribution.png      # Per-period bar + pie charts
│   ├── overall_balance.png         # Stacked bars + donut (all periods)
│   ├── hourly_heatmap.png          # Occupancy rate by hour × period
│   └── occupancy_timeline.png      # Rolling occupancy rate over time
├── 03_nan_analysis/
│   ├── nan_heatmap.png             # NaN fraction per sensor × period
│   ├── nan_bars_common.png         # NaN fraction for 15 common columns
│   └── nan_temporal.png            # NaN density over time
├── 04_time_coverage/
│   ├── gantt_chart.png             # Sensor vs label time range Gantt
│   ├── inter_period_gaps.png       # Gaps between collection periods
│   └── overlap_matrix.png          # Sensor-label overlap statistics
├── 05_sensor_timeline/
│   ├── core_dashboard_{P1-P4}.png  # 8 core sensors with occupancy overlay
│   └── all_sensors_{P1-P4}.png     # All sensors normalized overview
├── 06_correlation/
│   ├── correlation_matrix.png      # Sensor-sensor Pearson correlation
│   ├── sensor_label_correlation.png # Point-biserial with occupancy label
│   └── class_conditional_boxplots.png # Sensor distributions by class
└── 07_gap_analysis/
    ├── interval_histogram.png      # Time interval distribution
    ├── gap_detection.png           # Gap detection summary
    └── event_timing.png            # Inter-event time + hourly distribution
```

## Data Requirements

The toolkit expects paired sensor + label CSV files in the data root:

| File | Description |
|------|-------------|
| `merged_data_with_motion_count_0125_0126.csv` | P1 sensor data |
| `merged_data_with_motion_count_0201_0203.csv` | P2 sensor data |
| `merged_data_with_motion_count_0207_0209.csv` | P3 sensor data |
| `merged_data_with_motion_count_0209_0223.csv` | P4 sensor data |
| `occupancy_events_0126_processed.csv` | P1 labels |
| `occupancy_events_0201_0203_processed.csv` | P2 labels |
| `occupancy_events_0207_0209_processed.csv` | P3 labels |
| `occupancy_events_0210_0219_processed.csv` | P4 labels |

### Label format
```
time,Status,Head-count,At-home count
2026-02-10 07:37:00,LEAVE_HOME,1,1
```

### Sensor format
```
time,d620900d_motionSensor,408981c2_contactSensor,...
2026-02-09 23:20:00,0.0,0.0,...
```

## Dependencies

```
matplotlib >= 3.8
pandas >= 2.0
numpy >= 1.21
scipy >= 1.10
```

All dependencies are already included in the project's base requirements. No additional
installation needed.

## Key Insights Revealed

The analysis toolkit reveals several critical data characteristics:

1. **Class imbalance**: P3 has extreme 96:4 occupied:empty ratio vs P4's balanced 50:50
2. **Sensor schema inconsistency**: 21-34 columns per period, only 15 common across all
3. **NaN patterns**: Core sensors (motion, contact, temperature, power) have <15% NaN;
   window shade and non-essential sensors have >50% NaN
4. **Temporal patterns**: Clear daily rhythm — occupied at night (22h-07h), empty during
   work hours (09h-17h) — strongest in P4
5. **Discriminative sensors**: motionSensor (r=0.276, d=0.59) is the single best predictor;
   temperature sensors show negative correlation (occupied → lower temp)
6. **Perfect intervals**: All files have exact 5-minute binning with zero gaps
7. **Label coverage gap**: P4 has 1,116 unlabeled timesteps after last event (02/19-02/23)

## Module Architecture

```
analysis/
├── __init__.py              # Package exports
├── config.py                # Data paths, period defs, sensor mapping, colors
├── data_loader.py           # Unified data loading + event→label conversion
├── plot_01_overview.py      # Summary statistics and data inventory
├── plot_02_class_balance.py # Class distribution and temporal patterns
├── plot_03_nan_heatmap.py   # Missing value analysis
├── plot_04_time_coverage.py # Gantt chart and overlap analysis
├── plot_05_sensor_timeline.py # Sensor time series with label overlay
├── plot_06_correlation.py   # Correlation analysis and feature importance
├── plot_07_gap_analysis.py  # Time interval and event timing analysis
├── run_all.py               # Main entry point (CLI)
└── README.md                # This file
```
