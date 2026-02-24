"""Publication-quality style configuration for APC occupancy plots.

Provides consistent colors, fonts, and styling across all visualization
modules. Call ``setup_style()`` once at the start of a visualization
session to apply global matplotlib rcParams.
"""

from __future__ import annotations

import os

import matplotlib as mpl

# Use non-interactive backend for headless environments (GPU servers).
# Must be called before importing matplotlib.pyplot.
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colorblind-friendly binary palette (Okabe-Ito)
# ---------------------------------------------------------------------------
CLASS_COLORS: dict[int, str] = {
    0: "#0173B2",  # Blue  — Empty
    1: "#DE8F05",  # Orange — Occupied
}
CLASS_NAMES: dict[int, str] = {
    0: "Empty",
    1: "Occupied",
}

# Accent color for reference lines, EER markers, etc.
ACCENT_COLOR = "#CC3311"  # Red

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_ANNOTATION = 9

# ---------------------------------------------------------------------------
# Output defaults
# ---------------------------------------------------------------------------
DEFAULT_DPI = 300
DEFAULT_FORMATS = ("png", "pdf")
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_WIDE = (12, 4)
FIGSIZE_TALL = (5, 8)

# ---------------------------------------------------------------------------
# Active output configuration (set via configure_output())
# ---------------------------------------------------------------------------
_active_dpi: int = DEFAULT_DPI
_active_formats: tuple[str, ...] = DEFAULT_FORMATS


def configure_output(
    formats: list[str] | tuple[str, ...] | None = None,
    dpi: int | None = None,
) -> None:
    """Set output formats and DPI for all subsequent ``save_figure()`` calls.

    Parameters
    ----------
    formats : list or tuple of str, optional
        Output image formats (e.g. ``["png", "pdf"]``).
    dpi : int, optional
        Resolution for raster outputs.
    """
    global _active_dpi, _active_formats
    if formats is not None:
        _active_formats = tuple(formats)
    if dpi is not None:
        _active_dpi = dpi
        mpl.rcParams["figure.dpi"] = dpi
        mpl.rcParams["savefig.dpi"] = dpi


def setup_style() -> None:
    """Apply publication-quality matplotlib rcParams."""
    mpl.rcParams.update({
        "figure.dpi": _active_dpi,
        "savefig.dpi": _active_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        # Layout
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def save_figure(
    fig: plt.Figure,
    output_path,
    formats: tuple[str, ...] | None = None,
    dpi: int | None = None,
) -> None:
    """Save figure in multiple formats (e.g. PNG + PDF).

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    output_path : str or Path
        Base output path (extension is replaced per format).
    formats : tuple of str, optional
        Output formats. Defaults to the active configuration
        set by ``configure_output()`` (initially ``("png", "pdf")``).
    dpi : int, optional
        Resolution. Defaults to the active configuration.
    """
    from pathlib import Path

    if formats is None:
        formats = _active_formats
    if dpi is None:
        dpi = _active_dpi

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, format=fmt)
