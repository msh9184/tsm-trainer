"""Embedding-space augmentation for low-sample neural classification.

All augmentations operate on pre-extracted MantisV2 embedding tensors,
NOT on raw sensor data.  This avoids re-running the frozen backbone
and keeps augmentation in the fast, low-dimensional embedding space.

Strategies:
  1. GaussianNoise: additive N(0, sigma) noise
  2. Mixup: convex combination of embedding pairs + soft labels
  3. ChannelDrop: zero-out entire channel-level embedding segments

Designed for N~100 regimes where overfitting is the primary concern.
"""

from __future__ import annotations

import torch


def gaussian_noise(
    Z: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to embeddings.

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Input embeddings.
    sigma : float
        Standard deviation of the noise. Values 0.01–0.1 typical.
    generator : torch.Generator, optional
        For reproducible noise.

    Returns
    -------
    Tensor, shape (batch, embed_dim)
        Noisy embeddings (original Z is not modified).
    """
    if sigma <= 0:
        return Z
    noise = torch.randn(Z.shape, dtype=Z.dtype, device=Z.device, generator=generator)
    return Z + sigma * noise


def mixup(
    Z: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation: convex combination of embedding pairs.

    Generates virtual training samples by linearly interpolating between
    random pairs.  Returns soft label vectors (one-hot mixed).

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Input embeddings.
    y : Tensor, shape (batch,) of int64 or (batch, n_classes) of float
        Labels — integer class indices or already-soft probability vectors.
    alpha : float
        Beta distribution parameter. alpha=0 means no mixing, alpha=1 is
        uniform mixing.  Typical: 0.2–1.0.
    generator : torch.Generator, optional
        For reproducible shuffling and lambda sampling.

    Returns
    -------
    Z_mixed : Tensor, shape (batch, embed_dim)
    y_mixed : Tensor, shape (batch, n_classes) — soft labels
    """
    if alpha <= 0:
        # No mixing — return one-hot soft labels if integer labels provided
        if y.ndim == 1:
            n_classes = int(y.max().item()) + 1
            y_onehot = torch.zeros(len(y), n_classes, dtype=Z.dtype, device=Z.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            return Z, y_onehot
        return Z, y

    batch_size = Z.shape[0]

    # Convert integer labels to one-hot if needed
    if y.ndim == 1:
        n_classes = int(y.max().item()) + 1
        y_onehot = torch.zeros(batch_size, n_classes, dtype=Z.dtype, device=Z.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    else:
        y_onehot = y.clone()

    # Sample lambda from Beta(alpha, alpha)
    # PyTorch doesn't have Beta.sample with generator, so use numpy-style
    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample().item()

    # Random permutation for pairing
    if generator is not None:
        indices = torch.randperm(batch_size, generator=generator, device=Z.device)
    else:
        indices = torch.randperm(batch_size, device=Z.device)

    Z_mixed = lam * Z + (1 - lam) * Z[indices]
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[indices]

    return Z_mixed, y_mixed


def channel_drop(
    Z: torch.Tensor,
    n_channels: int,
    embed_per_channel: int,
    p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly zero-out full channel embedding segments.

    For multi-channel MantisV2 inputs, the embedding is a concatenation
    of per-channel embeddings.  This drops entire channel blocks with
    probability ``p``, forcing the head to not rely on any single sensor.

    Parameters
    ----------
    Z : Tensor, shape (batch, n_channels * embed_per_channel)
        Concatenated per-channel embeddings.
    n_channels : int
        Number of sensor channels.
    embed_per_channel : int
        Embedding dimension per channel.
    p : float
        Probability of dropping each channel block. Typical: 0.1–0.3.
    generator : torch.Generator, optional
        For reproducible drops.

    Returns
    -------
    Tensor, shape (batch, n_channels * embed_per_channel)
        Augmented embeddings (dropped channels zeroed).
    """
    if p <= 0 or n_channels <= 1:
        return Z

    batch_size = Z.shape[0]
    # Channel drop mask: (batch, n_channels)
    mask = torch.rand(
        batch_size, n_channels, dtype=Z.dtype, device=Z.device, generator=generator,
    ) >= p

    # Expand mask to embedding dimensions: (batch, n_channels * embed_per_channel)
    mask = mask.unsqueeze(-1).expand(-1, -1, embed_per_channel).reshape(batch_size, -1)

    return Z * mask


def apply_augmentation(
    Z: torch.Tensor,
    y: torch.Tensor,
    config: dict,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply configured augmentations to training embeddings.

    Augmentations are applied in order: channel_drop -> gaussian_noise -> mixup.
    Mixup returns soft labels; other augmentations preserve label format.

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Training embeddings.
    y : Tensor, shape (batch,) or (batch, n_classes)
        Training labels (integer or soft).
    config : dict
        Augmentation config with keys:
        - gaussian_noise_sigma (float): noise std, 0 = disabled
        - mixup_alpha (float): beta param, 0 = disabled
        - channel_drop_p (float): channel drop prob, 0 = disabled
        - n_channels (int): for channel_drop
        - embed_per_channel (int): for channel_drop
    generator : torch.Generator, optional

    Returns
    -------
    Z_aug : Tensor, shape (batch, embed_dim)
    y_aug : Tensor, shape (batch,) or (batch, n_classes)
    """
    # Channel drop (preserves labels)
    ch_drop_p = config.get("channel_drop_p", 0.0)
    if ch_drop_p > 0:
        n_ch = config.get("n_channels", 1)
        embed_per_ch = config.get("embed_per_channel", Z.shape[1])
        Z = channel_drop(Z, n_ch, embed_per_ch, ch_drop_p, generator)

    # Gaussian noise (preserves labels)
    sigma = config.get("gaussian_noise_sigma", 0.0)
    if sigma > 0:
        Z = gaussian_noise(Z, sigma, generator)

    # Mixup (produces soft labels)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    if mixup_alpha > 0:
        Z, y = mixup(Z, y, mixup_alpha, generator)

    return Z, y
