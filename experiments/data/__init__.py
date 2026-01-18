# Data package for experiments
"""
Data loading utilities for learned activations experiments:
- HuggingFace SatCLIP dataset
- Geographic data (elevation, population)
- Synthetic data (checkerboard)
"""

from .huggingface import SatCLIPHuggingFaceDataModule
from .geographic import ElevationDataset, PopulationDataset, GeographicDataModule
from .synthetic import CheckerboardDataset, SyntheticDataModule

__all__ = [
    "SatCLIPHuggingFaceDataModule",
    "ElevationDataset",
    "PopulationDataset",
    "GeographicDataModule",
    "CheckerboardDataset",
    "SyntheticDataModule",
]
