"""Publication-quality style configuration for enter/leave/stay plots.

Extended from apc_enter_leave to support 3-class and 5-class color palettes.
"""

from __future__ import annotations

import os

import matplotlib as mpl

if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palettes by label setting
# ---------------------------------------------------------------------------

# Setting 3 (3-class occupancy): Enter, Leave, Stay
CLASS_COLORS_3: dict[int, str] = {
    0: "#009E73",  # Green  - Enter
    1: "#CC3311",  # Red    - Leave
    2: "#0173B2",  # Blue   - Stay
}
CLASS_NAMES_3: dict[int, str] = {
    0: "Enter",
    1: "Leave",
    2: "Stay",
}

# Setting 1 (5-class)
CLASS_COLORS_5: dict[int, str] = {
    0: "#009E73",  # Green   - Enter_New
    1: "#56B4E9",  # LightBlue - Enter_Add
    2: "#CC3311",  # Red     - Leave_Last
    3: "#E69F00",  # Orange  - Leave_Reduce
    4: "#999999",  # Gray    - Stay
}
CLASS_NAMES_5: dict[int, str] = {
    0: "Enter_New",
    1: "Enter_Add",
    2: "Leave_Last",
    3: "Leave_Reduce",
    4: "Stay",
}

# Default: 3-class
CLASS_COLORS = CLASS_COLORS_3
CLASS_NAMES = CLASS_NAMES_3

ACCENT_COLOR = "#0173B2"

# ---------------------------------------------------------------------------
# Font sizes and output defaults
# ---------------------------------------------------------------------------
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_ANNOTATION = 9

DEFAULT_DPI = 300
DEFAULT_FORMATS = ("png", "pdf")
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_WIDE = (12, 4)
FIGSIZE_TALL = (5, 8)

_active_dpi: int = DEFAULT_DPI
_active_formats: tuple[str, ...] = DEFAULT_FORMATS


def configure_output(
    formats: list[str] | tuple[str, ...] | None = None,
    dpi: int | None = None,
) -> None:
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
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
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


def get_class_colors(n_classes: int) -> dict[int, str]:
    """Return appropriate color palette based on number of classes."""
    if n_classes <= 3:
        return CLASS_COLORS_3
    return CLASS_COLORS_5


def get_class_names_dict(n_classes: int) -> dict[int, str]:
    """Return class name dict based on number of classes."""
    if n_classes <= 3:
        return CLASS_NAMES_3
    return CLASS_NAMES_5
