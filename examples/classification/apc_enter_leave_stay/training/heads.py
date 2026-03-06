"""Neural classification heads for frozen MantisV2 embeddings.

Two head architectures: LinearHead (linear probe), MLPHead (multi-layer perceptron).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Linear probe: embed_dim -> n_classes."""

    def __init__(self, embed_dim: int, n_classes: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPHead(nn.Module):
    """Multi-layer perceptron classification head.

    Architecture per hidden layer:
        Linear -> BatchNorm1d (optional) -> GELU -> Dropout
    Followed by a final Linear projection to n_classes.
    """

    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(h)


HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "linear": LinearHead,
    "mlp": MLPHead,
}


def build_head(
    head_type: str,
    embed_dim: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    """Build a classification head by name."""
    cls = HEAD_REGISTRY.get(head_type)
    if cls is None:
        raise ValueError(
            f"Unknown head type: {head_type!r}. Available: {list(HEAD_REGISTRY)}"
        )
    return cls(embed_dim=embed_dim, n_classes=n_classes, **kwargs)
