# Activation functions package
"""
Modular activation functions for location encoders:
- ReLU (baseline)
- SIREN (sinusoidal)
- Spline (learned)
- RFF (Random Fourier Features)
"""

from .relu import ReLUActivation
from .siren import SIRENActivation, Sine
from .spline import SplineActivation
from .rff import RFFActivation

__all__ = [
    "ReLUActivation",
    "SIRENActivation",
    "Sine",
    "SplineActivation",
    "RFFActivation",
]

# Registry for easy config-based instantiation
ACTIVATION_REGISTRY = {
    "relu": ReLUActivation,
    "siren": SIRENActivation,
    "spline": SplineActivation,
    "rff": RFFActivation,
}


def get_activation(name: str, **kwargs):
    """Get activation class by name.

    Args:
        name: Activation name (relu, siren, spline, rff)
        **kwargs: Activation-specific parameters

    Returns:
        Instantiated activation module
    """
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATION_REGISTRY.keys())}")
    return ACTIVATION_REGISTRY[name](**kwargs)
