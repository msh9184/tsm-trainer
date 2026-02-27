"""Neural classification heads for frozen MantisV2 embeddings.

Four head architectures with increasing expressiveness:
  1. LinearHead     — simple linear probe baseline
  2. MLPHead        — multi-layer perceptron with BN + dropout
  3. MultiLayerFusionHead — learnable weighted sum of layer embeddings + MLP
  4. AttentionPoolHead    — cross-layer attention pooling + MLP

All heads operate on pre-extracted embedding vectors (not raw sensor data).
Designed for extreme low-sample regimes (N~100) with aggressive regularization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Linear probe: embed_dim -> n_classes.

    Establishes a linear separability baseline. No hidden layers,
    no nonlinearities.  ~embed_dim * n_classes parameters.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embedding vector.
    n_classes : int
        Number of output classes.
    """

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, embed_dim)

        Returns
        -------
        logits : Tensor, shape (batch, n_classes)
        """
        return self.fc(x)


class MLPHead(nn.Module):
    """Multi-layer perceptron classification head.

    Architecture per hidden layer:
        Linear -> BatchNorm1d (optional) -> GELU -> Dropout

    Followed by a final Linear projection to n_classes.

    Parameters
    ----------
    embed_dim : int
        Input embedding dimension.
    n_classes : int
        Number of output classes.
    hidden_dims : list[int]
        Sizes of hidden layers. E.g. [128, 64] for two hidden layers.
    dropout : float
        Dropout probability applied after each activation.
    use_batchnorm : bool
        Whether to apply BatchNorm1d before activation.
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
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, embed_dim)

        Returns
        -------
        logits : Tensor, shape (batch, n_classes)
        """
        h = self.features(x)
        return self.classifier(h)


class MultiLayerFusionHead(nn.Module):
    """Learnable weighted fusion of multiple transformer layer embeddings.

    Architecture:
        [L0, L1, ..., Ln] -> softmax(weights) -> weighted_sum -> MLPHead

    Learns which MantisV2 layers are most informative via a softmax-normalized
    weight vector.  The fused embedding is passed through an MLP classification
    head.

    Parameters
    ----------
    embed_dim : int
        Per-layer embedding dimension (all layers must have the same dim).
    n_layers : int
        Number of transformer layers to fuse.
    n_classes : int
        Number of output classes.
    hidden_dims : list[int]
        MLP hidden layer sizes after fusion.
    dropout : float
        Dropout probability in the MLP.
    use_batchnorm : bool
        Whether to use BatchNorm1d in the MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        # Learnable layer weights (initialized uniform via zeros -> softmax = 1/n)
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.mlp = MLPHead(
            embed_dim, n_classes,
            hidden_dims=hidden_dims, dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, layer_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass with multi-layer fusion.

        Parameters
        ----------
        layer_embeddings : list of Tensor
            Each tensor has shape (batch, embed_dim). One per layer.

        Returns
        -------
        logits : Tensor, shape (batch, n_classes)
        """
        # Stack: (n_layers, batch, embed_dim)
        stacked = torch.stack(layer_embeddings, dim=0)
        # Softmax weights: (n_layers, 1, 1) for broadcasting
        weights = F.softmax(self.layer_weights, dim=0).unsqueeze(-1).unsqueeze(-1)
        # Weighted sum: (batch, embed_dim)
        fused = (stacked * weights).sum(dim=0)
        return self.mlp(fused)

    def get_layer_weights(self) -> list[float]:
        """Return the current softmax-normalized layer weights."""
        with torch.no_grad():
            w = F.softmax(self.layer_weights, dim=0)
        return w.cpu().tolist()


class AttentionPoolHead(nn.Module):
    """Cross-layer attention pooling head.

    Architecture:
        stack(layers) -> MultiheadAttention(query=learned, kv=layers) -> MLP

    A single learned query token attends over the layer embeddings to produce
    a fixed-size representation. More expressive than weighted sum but higher
    overfitting risk with small N.

    Parameters
    ----------
    embed_dim : int
        Per-layer embedding dimension.
    n_layers : int
        Number of transformer layers (sequence length for attention).
    n_classes : int
        Number of output classes.
    n_heads : int
        Number of attention heads. Must divide embed_dim.
    hidden_dims : list[int]
        MLP hidden layer sizes after attention pooling.
    dropout : float
        Dropout probability in MLP and attention.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_classes: int,
        n_heads: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLPHead(
            embed_dim, n_classes,
            hidden_dims=hidden_dims, dropout=dropout,
            use_batchnorm=True,
        )

    def forward(self, layer_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass with attention pooling.

        Parameters
        ----------
        layer_embeddings : list of Tensor
            Each tensor has shape (batch, embed_dim). One per layer.

        Returns
        -------
        logits : Tensor, shape (batch, n_classes)
        """
        batch_size = layer_embeddings[0].shape[0]
        # Stack: (batch, n_layers, embed_dim)
        kv = torch.stack(layer_embeddings, dim=1)
        # Expand query: (batch, 1, embed_dim)
        q = self.query.expand(batch_size, -1, -1)
        # Attention: (batch, 1, embed_dim)
        attn_out, _ = self.attn(q, kv, kv)
        # Squeeze and normalize: (batch, embed_dim)
        pooled = self.layer_norm(attn_out.squeeze(1))
        return self.mlp(pooled)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "linear": LinearHead,
    "mlp": MLPHead,
    "multi_layer_fusion": MultiLayerFusionHead,
    "attention_pool": AttentionPoolHead,
}


def build_head(
    head_type: str,
    embed_dim: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    """Build a classification head by name.

    Parameters
    ----------
    head_type : str
        One of: "linear", "mlp", "multi_layer_fusion", "attention_pool".
    embed_dim : int
        Input embedding dimension.
    n_classes : int
        Number of output classes.
    **kwargs
        Additional arguments passed to the head constructor
        (e.g. hidden_dims, dropout, n_layers, n_heads).

    Returns
    -------
    nn.Module
    """
    cls = HEAD_REGISTRY.get(head_type)
    if cls is None:
        raise ValueError(
            f"Unknown head type: {head_type!r}. "
            f"Available: {list(HEAD_REGISTRY)}"
        )
    return cls(embed_dim=embed_dim, n_classes=n_classes, **kwargs)
