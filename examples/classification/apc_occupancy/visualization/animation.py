"""Training progression GIF animation for embedding visualization.

Tracks embeddings across fine-tuning epochs and creates an animated GIF
showing how the embedding space evolves during training.

Usage::

    tracker = EmbeddingTracker(output_dir=Path("results/snapshots"))
    for epoch in range(num_epochs):
        model.fit(X_train, y_train, num_epochs=1, ...)
        Z = model.transform(X_sample)
        tracker.save_snapshot(Z, y_sample, epoch=epoch)
    tracker.create_gif(fps=2)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .embeddings import reduce_dimensions
from .style import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_DPI,
    FIGSIZE_SINGLE,
    setup_style,
)

logger = logging.getLogger(__name__)


class EmbeddingTracker:
    """Track and animate embedding space evolution during training.

    Parameters
    ----------
    output_dir : Path or str
        Directory to save snapshot PNGs and the final GIF.
    method : str
        Dimensionality reduction method (``"tsne"`` or ``"pca"``).
        PCA is recommended for animation since it is deterministic and
        allows reusing the same fitted transform across epochs.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        output_dir: Path | str,
        method: str = "pca",
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.method = method
        self.random_state = random_state

        self._snapshots: list[Path] = []
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._axis_limits: tuple[float, float, float, float] | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        epoch: int,
    ) -> Path:
        """Reduce embeddings to 2D, save as PNG, and record in snapshot list.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(N, D)`` high-dimensional embeddings.
        labels : np.ndarray
            Shape ``(N,)`` class labels.
        epoch : int
            Current training epoch (used in filename and title).

        Returns
        -------
        Path
            Path to the saved snapshot PNG.
        """
        setup_style()

        if self.method == "pca":
            emb_2d = self._pca_transform(embeddings)
        else:
            emb_2d = reduce_dimensions(
                embeddings, method=self.method, random_state=self.random_state,
            )

        # Compute or reuse fixed axis limits
        if self._axis_limits is None:
            pad = 0.15  # Extra padding for future epochs
            x_range = emb_2d[:, 0].max() - emb_2d[:, 0].min()
            y_range = emb_2d[:, 1].max() - emb_2d[:, 1].min()
            self._axis_limits = (
                emb_2d[:, 0].min() - pad * x_range,
                emb_2d[:, 0].max() + pad * x_range,
                emb_2d[:, 1].min() - pad * y_range,
                emb_2d[:, 1].max() + pad * y_range,
            )

        # Plot
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        for cls in sorted(CLASS_COLORS):
            mask = labels == cls
            if not mask.any():
                continue
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls],
                s=15,
                alpha=0.6,
                edgecolors="none",
            )

        method_label = self.method.upper() if self.method != "tsne" else "t-SNE"
        ax.set_title(f"Epoch {epoch} ({method_label})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=2, loc="upper right")

        x_min, x_max, y_min, y_max = self._axis_limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        snapshot_path = self.output_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(snapshot_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

        self._snapshots.append(snapshot_path)
        logger.debug("Saved embedding snapshot: %s", snapshot_path)
        return snapshot_path

    def _pca_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA reduction, fitting on first call and reusing afterwards."""
        if self._scaler is None:
            self._scaler = StandardScaler().fit(embeddings)
        X = self._scaler.transform(embeddings)

        if self._pca is None:
            n_components = min(2, X.shape[1], X.shape[0])
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            self._pca.fit(X)

        result = self._pca.transform(X)
        # Pad to (N, 2) if fewer than 2 components were available
        if result.shape[1] < 2:
            result = np.hstack([result, np.zeros((result.shape[0], 2 - result.shape[1]))])
        return result

    def create_gif(
        self,
        output_path: Path | str | None = None,
        fps: int = 2,
        duration_per_frame_ms: int | None = None,
    ) -> Path | None:
        """Combine saved snapshot PNGs into an animated GIF.

        Parameters
        ----------
        output_path : Path or str, optional
            Output GIF path. Defaults to ``{output_dir}/training_progression.gif``.
        fps : int
            Frames per second (ignored if *duration_per_frame_ms* is set).
        duration_per_frame_ms : int, optional
            Duration of each frame in milliseconds. Overrides *fps*.

        Returns
        -------
        Path or None
            Path to the created GIF, or None if no snapshots exist.
        """
        if not self._snapshots:
            logger.warning("No snapshots to create GIF from")
            return None

        try:
            import imageio
        except ImportError:
            logger.warning(
                "imageio not installed, cannot create GIF. "
                "Install with: pip install imageio"
            )
            return None

        if output_path is None:
            output_path = self.output_dir / "training_progression.gif"
        output_path = Path(output_path)

        if duration_per_frame_ms is not None:
            duration_s = duration_per_frame_ms / 1000.0
        else:
            duration_s = 1.0 / fps

        frames = []
        for snap_path in self._snapshots:
            frame = imageio.imread(str(snap_path))
            frames.append(frame)

        # Hold last frame longer
        for _ in range(3):
            frames.append(frames[-1])

        # mimwrite() works in both imageio 2.x and 3.x
        imageio.mimwrite(str(output_path), frames, duration=duration_s, loop=0)

        logger.info(
            "Created GIF: %s (%d frames, %.0f ms/frame)",
            output_path, len(self._snapshots), duration_s * 1000,
        )
        return output_path

    @property
    def n_snapshots(self) -> int:
        """Number of saved snapshots."""
        return len(self._snapshots)
