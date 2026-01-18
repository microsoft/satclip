# Positional Encodings package
"""
Modular positional encodings for geographic coordinates:
- Spherical Harmonics (SH) - the SatCLIP default
- Raw coordinates (normalized lon/lat)
- RFF encoding (Random Fourier Features)
- Cartesian 3D (unit sphere projection)
"""

from .spherical_harmonics import SphericalHarmonicsEncoding
from .raw import RawCoordinateEncoding
from .rff_encoding import RFFEncoding
from .cartesian import Cartesian3DEncoding

__all__ = [
    "SphericalHarmonicsEncoding",
    "RawCoordinateEncoding",
    "RFFEncoding",
    "Cartesian3DEncoding",
]

# Registry for config-based instantiation
ENCODING_REGISTRY = {
    "spherical_harmonics": SphericalHarmonicsEncoding,
    "sh": SphericalHarmonicsEncoding,  # alias
    "raw": RawCoordinateEncoding,
    "rff": RFFEncoding,
    "cartesian3d": Cartesian3DEncoding,
}


def get_encoding(name: str, **kwargs):
    """Get positional encoding by name.

    Args:
        name: Encoding name (spherical_harmonics, sh, raw, rff, cartesian3d)
        **kwargs: Encoding-specific parameters

    Returns:
        Instantiated encoding module
    """
    if name not in ENCODING_REGISTRY:
        raise ValueError(f"Unknown encoding: {name}. Available: {list(ENCODING_REGISTRY.keys())}")
    return ENCODING_REGISTRY[name](**kwargs)
