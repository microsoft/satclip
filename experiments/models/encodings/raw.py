"""Raw Coordinate Encoding.

The simplest encoding: just normalize lon/lat to a reasonable range.

From experiments:
- Raw + ReLU/Spline can work, but generally worse than SH
- Raw + RFF (correct architecture) is competitive
- Use for ablation studies or when SH is too heavy
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RawCoordinateEncoding(nn.Module):
    """Raw coordinate encoding with optional normalization.

    Simply normalizes lon/lat to a fixed range (default: [-1, 1]).
    Output dimension is always 2.
    """

    def __init__(
        self,
        normalize: bool = True,
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        lat_range: Tuple[float, float] = (-90.0, 90.0),
        output_range: Tuple[float, float] = (-1.0, 1.0),
        **kwargs,
    ):
        """Initialize raw coordinate encoding.

        Args:
            normalize: Whether to normalize coordinates
            lon_range: Expected input longitude range
            lat_range: Expected input latitude range
            output_range: Desired output range after normalization
            **kwargs: Ignored
        """
        super().__init__()
        self.normalize = normalize
        self.embedding_dim = 2

        # Register normalization parameters as buffers
        self.register_buffer(
            "lon_offset", torch.tensor((lon_range[0] + lon_range[1]) / 2)
        )
        self.register_buffer(
            "lon_scale", torch.tensor((lon_range[1] - lon_range[0]) / 2)
        )
        self.register_buffer(
            "lat_offset", torch.tensor((lat_range[0] + lat_range[1]) / 2)
        )
        self.register_buffer(
            "lat_scale", torch.tensor((lat_range[1] - lat_range[0]) / 2)
        )
        self.register_buffer(
            "output_scale",
            torch.tensor((output_range[1] - output_range[0]) / 2),
        )
        self.register_buffer(
            "output_offset",
            torch.tensor((output_range[0] + output_range[1]) / 2),
        )

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """Encode coordinates.

        Args:
            lonlat: Tensor of shape (batch, 2) with (longitude, latitude)

        Returns:
            Encoded coordinates of shape (batch, 2)
        """
        if not self.normalize:
            return lonlat

        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # Normalize to [-1, 1]
        lon_norm = (lon - self.lon_offset) / self.lon_scale
        lat_norm = (lat - self.lat_offset) / self.lat_scale

        # Scale to output range
        lon_out = lon_norm * self.output_scale + self.output_offset
        lat_out = lat_norm * self.output_scale + self.output_offset

        return torch.stack([lon_out, lat_out], dim=-1)

    @property
    def output_dim(self) -> int:
        """Output dimension of encoding."""
        return self.embedding_dim

    def extra_repr(self) -> str:
        return f"normalize={self.normalize}, dim={self.embedding_dim}"


class DegreeToRadianEncoding(nn.Module):
    """Convert degrees to radians, normalized to [-pi, pi].

    This is what SIREN networks typically expect.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_dim = 2

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """Convert degrees to radians.

        Args:
            lonlat: Tensor of shape (batch, 2) with (lon, lat) in degrees

        Returns:
            Tensor of shape (batch, 2) in radians, range [-pi, pi]
        """
        return torch.deg2rad(lonlat) - torch.pi

    @property
    def output_dim(self) -> int:
        return self.embedding_dim
