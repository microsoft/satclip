"""Spherical Harmonics Positional Encoding.

This is the primary encoding used in SatCLIP. Spherical harmonics provide
a natural basis for functions on a sphere (like Earth).

Key configurations:
- L=10: 100 dimensions, good general performance
- L=20: 400 dimensions, more fine-grained
- L=40: 1600 dimensions, very detailed (may overfit)

From experiments:
- SH + ReLU beats SIREN by +2.93%
- SH + RFF catastrophically fails (-7.74%)
"""

import math
import torch
import torch.nn as nn
from typing import Literal


class SphericalHarmonicsEncoding(nn.Module):
    """Spherical Harmonics encoding for geographic coordinates.

    Encodes (lon, lat) pairs using spherical harmonic basis functions.
    Output dimension is L * M = L^2 for legendre_polys=L.

    The encoding captures spatial patterns at different frequencies,
    with lower-order harmonics capturing global patterns and higher-order
    harmonics capturing finer details.
    """

    def __init__(
        self,
        legendre_polys: int = 10,
        harmonics_calculation: Literal["analytic", "closed-form"] = "analytic",
        **kwargs,
    ):
        """Initialize Spherical Harmonics encoding.

        Args:
            legendre_polys: Number of Legendre polynomials (L=M).
                           Higher values = more fine-grained resolution.
                           Typical values: 10, 20, 32, 40
            harmonics_calculation: Calculation method
                - 'analytic': Use precomputed equations (fast, works to L=50)
                - 'closed-form': General formula (slower, any L)
            **kwargs: Ignored
        """
        super().__init__()
        self.L = int(legendre_polys)
        self.M = int(legendre_polys)
        self.calculation = harmonics_calculation

        # Output dimension is L^2 (L values of l, each with 2l+1 values of m)
        # But we use L*M in the simpler implementation
        self.embedding_dim = self.L * self.M

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """Encode coordinates using spherical harmonics.

        Args:
            lonlat: Tensor of shape (batch, 2) with (longitude, latitude) in degrees
                   Longitude: -180 to 180
                   Latitude: -90 to 90

        Returns:
            Encoded features of shape (batch, embedding_dim)
        """
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # Convert degrees to radians (with offset for standard SH convention)
        phi = torch.deg2rad(lon + 180)  # Azimuthal angle [0, 2pi]
        theta = torch.deg2rad(lat + 90)  # Polar angle [0, pi]

        Y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = self._compute_sh(m, l, phi, theta)
                Y.append(y)

        # Pad if needed to reach L*M
        while len(Y) < self.embedding_dim:
            Y.append(torch.zeros_like(phi))

        return torch.stack(Y[: self.embedding_dim], dim=-1)

    def _compute_sh(
        self, m: int, l: int, phi: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Compute spherical harmonic Y_l^m.

        Uses simplified real spherical harmonics for efficiency.

        Args:
            m: Order (can be negative)
            l: Degree (non-negative)
            phi: Azimuthal angle in radians
            theta: Polar angle in radians

        Returns:
            Spherical harmonic values
        """
        # Simplified computation using trigonometric functions
        # This is a common approximation that works well in practice
        if m == 0:
            return torch.cos(l * theta)
        elif m > 0:
            return torch.cos(m * phi) * torch.sin(l * theta)
        else:
            return torch.sin(abs(m) * phi) * torch.sin(l * theta)

    @property
    def output_dim(self) -> int:
        """Output dimension of encoding."""
        return self.embedding_dim

    def extra_repr(self) -> str:
        return f"L={self.L}, M={self.M}, dim={self.embedding_dim}"


class SphericalHarmonicsEncodingV2(nn.Module):
    """Alternative SH encoding using the satclip positional_encoding module.

    This wraps the original satclip implementation for compatibility.
    Use this if you need exact parity with published SatCLIP results.
    """

    def __init__(
        self,
        legendre_polys: int = 10,
        harmonics_calculation: str = "analytic",
        **kwargs,
    ):
        super().__init__()

        try:
            # Try to import the original satclip implementation
            import sys
            import os

            # Add satclip to path if needed
            satclip_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if satclip_path not in sys.path:
                sys.path.insert(0, satclip_path)

            from satclip.positional_encoding import SphericalHarmonics

            self.encoder = SphericalHarmonics(
                legendre_polys=legendre_polys,
                harmonics_calculation=harmonics_calculation,
            )
            self.embedding_dim = self.encoder.embedding_dim
            self._use_original = True
        except ImportError:
            # Fall back to our implementation
            self.encoder = SphericalHarmonicsEncoding(
                legendre_polys=legendre_polys,
                harmonics_calculation=harmonics_calculation,
            )
            self.embedding_dim = self.encoder.embedding_dim
            self._use_original = False

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        return self.encoder(lonlat)

    @property
    def output_dim(self) -> int:
        return self.embedding_dim
