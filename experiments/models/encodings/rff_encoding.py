"""Random Fourier Features Encoding for coordinates.

RFF provides a fixed random projection of coordinates into a
higher-dimensional feature space.

IMPORTANT NOTES:
1. Do NOT combine RFF encoding with SH encoding - they interfere
2. Use RFF -> Linear (no intermediate activation)
3. This is for encoding coordinates, not as an activation
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class RFFEncoding(nn.Module):
    """Random Fourier Features encoding for geographic coordinates.

    Projects coordinates using random frequencies:
    phi(x) = [sin(2*pi*Bx), cos(2*pi*Bx)]

    This approximates shift-invariant kernels and can capture
    high-frequency spatial patterns.
    """

    def __init__(
        self,
        input_dim: int = 2,
        n_features: int = 256,
        sigma: float = 10.0,
        normalize_input: bool = True,
        coord_scale: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize RFF encoding.

        Args:
            input_dim: Input dimension (2 for lon/lat)
            n_features: Number of output features (will be this exact number,
                       using n_features/2 sin + n_features/2 cos)
            sigma: Scale of random frequencies
                  - Higher = captures higher frequency patterns
                  - Typical: 1.0 to 100.0
            normalize_input: Whether to normalize input coordinates
            coord_scale: Scale for normalization (default: [180, 90] for lon/lat)
            seed: Random seed for reproducibility
            **kwargs: Ignored
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = sigma
        self.normalize_input = normalize_input
        self.embedding_dim = n_features

        if coord_scale is None:
            coord_scale = torch.tensor([180.0, 90.0])
        self.register_buffer("coord_scale", coord_scale)

        # Generate random frequency matrix
        if seed is not None:
            torch.manual_seed(seed)

        B = torch.randn(input_dim, n_features // 2) * sigma
        self.register_buffer("B", B)

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """Encode coordinates using RFF.

        Args:
            lonlat: Tensor of shape (batch, 2) with (lon, lat)

        Returns:
            RFF features of shape (batch, n_features)
        """
        x = lonlat

        if self.normalize_input:
            x = x / self.coord_scale.to(x.device)

        # Project: (batch, input_dim) @ (input_dim, n_features/2)
        x_proj = 2 * math.pi * x @ self.B

        # Concatenate sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    @property
    def output_dim(self) -> int:
        """Output dimension of encoding."""
        return self.embedding_dim

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"n_features={self.n_features}, "
            f"sigma={self.sigma}"
        )


class MultiScaleRFFEncoding(nn.Module):
    """Multi-scale RFF encoding with multiple sigma values.

    Combines RFF at multiple scales to capture patterns at
    different spatial frequencies.
    """

    def __init__(
        self,
        input_dim: int = 2,
        n_features_per_scale: int = 64,
        sigmas: tuple = (1.0, 5.0, 10.0, 50.0),
        normalize_input: bool = True,
        **kwargs,
    ):
        """Initialize multi-scale RFF.

        Args:
            input_dim: Input dimension
            n_features_per_scale: Features per sigma value
            sigmas: Tuple of sigma values (scales)
            normalize_input: Whether to normalize inputs
            **kwargs: Ignored
        """
        super().__init__()
        self.sigmas = sigmas
        self.n_features_per_scale = n_features_per_scale
        self.embedding_dim = len(sigmas) * n_features_per_scale

        self.encoders = nn.ModuleList(
            [
                RFFEncoding(
                    input_dim=input_dim,
                    n_features=n_features_per_scale,
                    sigma=s,
                    normalize_input=normalize_input,
                    seed=i,  # Different random features for each scale
                )
                for i, s in enumerate(sigmas)
            ]
        )

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """Encode at multiple scales and concatenate.

        Args:
            lonlat: Input coordinates

        Returns:
            Concatenated multi-scale features
        """
        features = [enc(lonlat) for enc in self.encoders]
        return torch.cat(features, dim=-1)

    @property
    def output_dim(self) -> int:
        return self.embedding_dim
