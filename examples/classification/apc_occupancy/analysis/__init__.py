"""APC Occupancy Data Analysis & EDA Toolkit.

Comprehensive exploratory data analysis for SmartThings sensor data
used in occupancy detection. Generates publication-quality visualizations
including:

  - Dataset overview and summary statistics
  - Class balance / imbalance analysis
  - Missing value (NaN) heatmaps
  - Sensor-label time coverage and overlap (Gantt chart)
  - Sensor time series with occupancy label overlay
  - Sensor correlation analysis
  - Time interval gap analysis

Usage (from GPU server)::

    python -m examples.classification.apc_occupancy.analysis.run_all \\
        --data-root /group-volume/workspace/haeri.kim/Time-Series/data/SmartThings/Samsung_QST_Data/enter_leave/ \\
        --output-dir results/apc_analysis/

Usage (from WSL with local sample data)::

    python -m examples.classification.apc_occupancy.analysis.run_all \\
        --data-root docs/sample_downstream_task/ \\
        --output-dir results/apc_analysis/
"""
