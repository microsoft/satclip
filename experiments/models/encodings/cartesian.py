"""Cartesian 3D Encoding for geographic coordinates.

Projects lon/lat to 3D unit sphere coordinates:
(lon, lat) -> (x, y, z) where x^2 + y^2 + z^2 = 1

This preserves distances and is natural for global data.
"""

import torch
import torch.nn as nn
import math


class Cartesian3DEncoding(nn.Module):
    """Project geographic coordinates to 3D Cartesian on unit sphere.

    Converts (longitude, latitude) to (x, y, z) on the unit sphere:
    - x = cos(lon) * cos(lat)
    - y = sin(lon) * cos(lat)
    - z = sin(lat)

    Output dimension is 3.
    """

    def __init__(self, **kwargs):
        """Initialize Cartesian 3D encoding.

        Args:
            **kwargs: Ignored
        """
        super().__init__()
        self.embedding_dim = 3

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """Project to 3D Cartesian coordinates.

        Args:
            lonlat: Tensor of shape (batch, 2) with (lon, lat) in degrees

        Returns:
            3D coordinates of shape (batch, 3)
        """
        lon_rad = torch.deg2rad(lonlat[:, 0])
        lat_rad = torch.deg2rad(lonlat[:, 1])

        cos_lat = torch.cos(lat_rad)

        x = torch.cos(lon_rad) * cos_lat
        y = torch.sin(lon_rad) * cos_lat
        z = torch.sin(lat_rad)

        return torch.stack([x, y, z], dim=-1)

    @property
    def output_dim(self) -> int:
        """Output dimension of encoding."""
        return self.embedding_dim

    def extra_repr(self) -> str:
        return f"dim={self.embedding_dim}"


class GridEncoding(nn.Module):
    """2D sinusoidal grid encoding (similar to Transformer positional encoding).

    Uses multiple frequencies for both lon and lat:
    [sin(f1*lon), cos(f1*lon), sin(f1*lat), cos(f1*lat), ...]
    """

    def __init__(
        self,
        frequency_num: int = 16,
        min_radius: float = 0.00001,
        max_radius: float = 0.01,
        **kwargs,
    ):
        """Initialize grid encoding.

        Args:
            frequency_num: Number of frequency bands
            min_radius: Minimum frequency
            max_radius: Maximum frequency
            **kwargs: Ignored
        """
        super().__init__()
        self.frequency_num = frequency_num
        self.embedding_dim = frequency_num * 4  # sin/cos for both lon/lat

        # Create frequency bands (geometric spacing)
        freqs = torch.exp(
            torch.linspace(
                math.log(min_radius),
                math.log(max_radius),
                frequency_num,
            )
        )
        self.register_buffer("freqs", freqs)

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """Apply grid encoding.

        Args:
            lonlat: Tensor of shape (batch, 2)

        Returns:
            Encoded features of shape (batch, frequency_num * 4)
        """
        lon, lat = lonlat[:, 0:1], lonlat[:, 1:2]

        # Scale by frequencies
        lon_scaled = lon * self.freqs  # (batch, frequency_num)
        lat_scaled = lat * self.freqs  # (batch, frequency_num)

        # Apply sin/cos
        features = torch.cat(
            [
                torch.sin(lon_scaled),
                torch.cos(lon_scaled),
                torch.sin(lat_scaled),
                torch.cos(lat_scaled),
            ],
            dim=-1,
        )

        return features

    @property
    def output_dim(self) -> int:
        return self.embedding_dim
