"""Embedding space visualization with t-SNE, UMAP, and PCA.

Provides dimensionality reduction and scatter-plot visualizations for
high-dimensional MantisV2 embeddings. UMAP is optional — if ``umap-learn``
is not installed, the module falls back to t-SNE with a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .style import (
    CLASS_COLORS,
    CLASS_NAMES,
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    save_figure,
    setup_style,
)

logger = logging.getLogger(__name__)

_UMAP_AVAILABLE = False
try:
    from umap import UMAP  # type: ignore[import-untyped]

    _UMAP_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    pca_components: int = 50,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D for visualization.

    Pipeline: ``StandardScaler -> PCA(D->min(D,pca_components)) -> method(->2)``.
    """
    n_samples, n_features = embeddings.shape

    if n_samples < 2:
        logger.warning("Only %d sample(s); returning zero-padded 2D output", n_samples)
        return np.zeros((n_samples, 2), dtype=np.float64)

    X = StandardScaler().fit_transform(embeddings)

    if method == "pca":
        pca = PCA(n_components=min(2, n_features, n_samples), random_state=random_state)
        result = pca.fit_transform(X)
        if result.shape[1] < 2:
            result = np.hstack([result, np.zeros((n_samples, 2 - result.shape[1]))])
        return result

    n_pre = min(pca_components, n_features, n_samples)
    if n_features > n_pre:
        X = PCA(n_components=n_pre, random_state=random_state).fit_transform(X)

    if method == "tsne":
        perplexity = kwargs.pop("perplexity", min(30, max(1, n_samples - 1)))
        max_iter = kwargs.pop("max_iter", kwargs.pop("n_iter", 1000))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )
        return tsne.fit_transform(X)

    if method == "umap":
        if not _UMAP_AVAILABLE:
            logger.warning(
                "umap-learn not installed, falling back to t-SNE. "
                "Install with: pip install umap-learn"
            )
            return reduce_dimensions(
                embeddings, method="tsne",
                pca_components=pca_components, random_state=random_state,
            )
        n_neighbors = kwargs.pop("n_neighbors", min(15, max(2, n_samples - 1)))
        min_dist = kwargs.pop("min_dist", 0.1)
        metric = kwargs.pop("metric", "cosine")
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **kwargs,
        )
        return reducer.fit_transform(X)

    raise ValueError(f"Unknown reduction method: {method!r}. Use 'tsne', 'umap', or 'pca'.")


def reduce_dimensions_joint(
    *arrays: np.ndarray,
    method: str = "tsne",
    pca_components: int = 50,
    random_state: int = 42,
    **kwargs,
) -> list[np.ndarray]:
    """Reduce multiple arrays in a shared coordinate space."""
    sizes = [len(a) for a in arrays]
    combined = np.concatenate(arrays, axis=0)
    reduced = reduce_dimensions(
        combined, method=method, pca_components=pca_components,
        random_state=random_state, **kwargs,
    )
    result = []
    offset = 0
    for s in sizes:
        result.append(reduced[offset : offset + s])
        offset += s
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    split: str = "test",
    method: str = "tsne",
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of 2D embeddings colored by class label."""
    setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    for cls in sorted(CLASS_COLORS):
        mask = labels == cls
        if not mask.any():
            continue
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            s=25,
            alpha=0.7,
            edgecolors="none",
        )

    if not title:
        method_upper = method.upper() if method != "tsne" else "t-SNE"
        title = f"{method_upper} — {split.capitalize()} Embeddings"
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig, ax


def plot_embeddings_multi_method(
    embeddings: np.ndarray,
    labels: np.ndarray,
    methods: list[str] | None = None,
    title: str = "",
    output_path: Path | str | None = None,
    random_state: int = 42,
) -> plt.Figure:
    """Side-by-side comparison of PCA, t-SNE, and (optionally) UMAP."""
    setup_style()
    if methods is None:
        methods = ["pca", "tsne"]
        if _UMAP_AVAILABLE:
            methods.append("umap")

    methods = [m for m in methods if m != "umap" or _UMAP_AVAILABLE]
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        emb_2d = reduce_dimensions(embeddings, method=method, random_state=random_state)
        method_label = method.upper() if method != "tsne" else "t-SNE"
        plot_embeddings(emb_2d, labels, title=method_label, method=method, ax=ax)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig
