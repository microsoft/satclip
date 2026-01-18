"""Random Fourier Features (RFF) Activation.

IMPORTANT: RFF + SH (Spherical Harmonics) catastrophically fails!
From NB16-17: RFF + SH gives -7.74% vs baseline due to frequency interference.

RFF should be used:
- With raw coordinates (not SH-encoded)
- With direct linear output (no intermediate activation)

The "corrected" RFF architecture (from NB21f):
- RFF encoding -> Linear output (no ReLU in between)
- This preserves the kernel approximation property
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class RFFActivation(nn.Module):
    """Random Fourier Features as activation/encoding.

    RFF approximates a shift-invariant kernel using random projections:
    k(x, y) ≈ <phi(x), phi(y)>
    where phi(x) = [cos(Bx), sin(Bx)]

    WARNING: Do NOT use with Spherical Harmonics encoding!
    The frequency representations conflict and cause catastrophic failure.
    """

    def __init__(
        self,
        input_dim: int = 2,
        n_features: int = 256,
        sigma: float = 10.0,
        learnable: bool = False,
        **kwargs,
    ):
        """Initialize RFF activation.

        Args:
            input_dim: Input dimension (typically 2 for lon/lat)
            n_features: Number of random features (output will be n_features)
            sigma: Scale for random frequencies (higher = higher frequency features)
            learnable: Whether frequencies are learnable (not recommended, from NB17)
            **kwargs: Ignored
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = sigma
        self.learnable = learnable

        # Random frequency matrix: output_dim/2 because we concatenate sin and cos
        B = torch.randn(input_dim, n_features // 2) * sigma

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RFF encoding.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            RFF features of shape (..., n_features)
        """
        # Project input: x @ B gives shape (..., n_features/2)
        x_proj = 2 * math.pi * x @ self.B

        # Concatenate sin and cos features
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    @property
    def output_dim(self) -> int:
        """Output dimension of RFF features."""
        return self.n_features

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"n_features={self.n_features}, "
            f"sigma={self.sigma}, "
            f"learnable={self.learnable}"
        )


class RFFModel(nn.Module):
    """Complete RFF model with correct architecture.

    From NB21f: The correct RFF architecture is:
    1. Normalize coordinates
    2. Apply RFF encoding (sin/cos)
    3. Direct linear output (NO intermediate activation!)

    This preserves the kernel approximation property that makes RFF work.
    """

    def __init__(
        self,
        input_dim: int = 2,
        n_features: int = 256,
        output_dim: int = 1,
        sigma: float = 10.0,
        normalize_coords: bool = True,
        coord_scale: Optional[torch.Tensor] = None,
    ):
        """Initialize correct RFF model.

        Args:
            input_dim: Input dimension
            n_features: Number of RFF features
            output_dim: Output dimension
            sigma: RFF frequency scale
            normalize_coords: Whether to normalize input coordinates
            coord_scale: Optional scale for coordinate normalization
                        Default: [180, 90] for lon/lat
        """
        super().__init__()
        self.normalize_coords = normalize_coords

        if coord_scale is None:
            coord_scale = torch.tensor([180.0, 90.0])
        self.register_buffer("coord_scale", coord_scale)

        self.rff = RFFActivation(
            input_dim=input_dim,
            n_features=n_features,
            sigma=sigma,
            learnable=False,  # Don't use learnable, it's catastrophic
        )

        # Direct linear output - NO activation after RFF!
        self.output = nn.Linear(n_features, output_dim)
        nn.init.kaiming_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with correct RFF architecture.

        Args:
            x: Input coordinates of shape (batch, input_dim)

        Returns:
            Output of shape (batch, output_dim)
        """
        if self.normalize_coords:
            x = x / self.coord_scale.to(x.device)

        # RFF encoding
        x = self.rff(x)

        # Direct linear output
        return self.output(x)
