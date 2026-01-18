"""Synthetic datasets for controlled experiments.

Includes:
- Checkerboard patterns (test spatial frequency learning)
- Gaussian mixtures
- Custom function fitting
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from typing import Optional, Tuple, Callable, Literal
import numpy as np
import math


class CheckerboardDataset(Dataset):
    """Checkerboard pattern dataset for testing high-frequency learning.

    Creates a checkerboard pattern on the globe where the value
    alternates between 0 and 1 based on grid cells.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        grid_size: float = 10.0,
        noise: float = 0.0,
        region: Optional[Tuple[float, float, float, float]] = None,
        seed: int = 42,
    ):
        """Initialize checkerboard dataset.

        Args:
            n_samples: Number of samples
            grid_size: Size of checkerboard squares in degrees
            noise: Standard deviation of Gaussian noise to add
            region: Optional bounding box (lon_min, lon_max, lat_min, lat_max)
            seed: Random seed
        """
        self.grid_size = grid_size
        self.noise = noise

        np.random.seed(seed)

        # Sample coordinates
        if region is None:
            region = (-180, 180, -90, 90)

        lon_min, lon_max, lat_min, lat_max = region

        lons = np.random.uniform(lon_min, lon_max, n_samples)
        lats = np.random.uniform(lat_min, lat_max, n_samples)

        # Compute checkerboard values
        lon_idx = np.floor(lons / grid_size).astype(int)
        lat_idx = np.floor(lats / grid_size).astype(int)
        values = ((lon_idx + lat_idx) % 2).astype(np.float32)

        # Add noise
        if noise > 0:
            values = values + np.random.normal(0, noise, n_samples)
            values = np.clip(values, 0, 1)

        self.coords = torch.from_numpy(
            np.column_stack([lons, lats])
        ).float()
        self.values = torch.from_numpy(values).float()

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict:
        return {
            "coords": self.coords[idx],
            "target": self.values[idx],
        }


class GaussianMixtureDataset(Dataset):
    """Gaussian mixture model dataset.

    Creates a spatial pattern with Gaussian bumps at random centers.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_centers: int = 10,
        sigma: float = 10.0,
        region: Optional[Tuple[float, float, float, float]] = None,
        seed: int = 42,
    ):
        """Initialize Gaussian mixture dataset.

        Args:
            n_samples: Number of samples
            n_centers: Number of Gaussian centers
            sigma: Standard deviation of Gaussians in degrees
            region: Optional bounding box
            seed: Random seed
        """
        np.random.seed(seed)

        if region is None:
            region = (-180, 180, -90, 90)

        lon_min, lon_max, lat_min, lat_max = region

        # Sample random centers
        center_lons = np.random.uniform(lon_min, lon_max, n_centers)
        center_lats = np.random.uniform(lat_min, lat_max, n_centers)

        # Sample data points
        lons = np.random.uniform(lon_min, lon_max, n_samples)
        lats = np.random.uniform(lat_min, lat_max, n_samples)

        # Compute values as sum of Gaussians
        values = np.zeros(n_samples)
        for i in range(n_centers):
            dist_sq = (lons - center_lons[i]) ** 2 + (lats - center_lats[i]) ** 2
            values += np.exp(-dist_sq / (2 * sigma ** 2))

        # Normalize to [0, 1]
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)

        self.coords = torch.from_numpy(
            np.column_stack([lons, lats])
        ).float()
        self.values = torch.from_numpy(values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict:
        return {
            "coords": self.coords[idx],
            "target": self.values[idx],
        }


class FunctionFittingDataset(Dataset):
    """Dataset for fitting arbitrary functions of coordinates.

    Allows testing on custom target functions.
    """

    def __init__(
        self,
        target_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_samples: int = 10000,
        region: Optional[Tuple[float, float, float, float]] = None,
        noise: float = 0.0,
        seed: int = 42,
    ):
        """Initialize function fitting dataset.

        Args:
            target_fn: Function (lon, lat) -> values
            n_samples: Number of samples
            region: Optional bounding box
            noise: Noise standard deviation
            seed: Random seed
        """
        np.random.seed(seed)

        if region is None:
            region = (-180, 180, -90, 90)

        lon_min, lon_max, lat_min, lat_max = region

        lons = np.random.uniform(lon_min, lon_max, n_samples)
        lats = np.random.uniform(lat_min, lat_max, n_samples)

        values = target_fn(lons, lats)

        if noise > 0:
            values = values + np.random.normal(0, noise, n_samples)

        self.coords = torch.from_numpy(
            np.column_stack([lons, lats])
        ).float()
        self.values = torch.from_numpy(values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict:
        return {
            "coords": self.coords[idx],
            "target": self.values[idx],
        }


# Predefined target functions for testing
def sinusoidal_function(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Multi-frequency sinusoidal pattern."""
    return (
        np.sin(lon * np.pi / 30) * np.cos(lat * np.pi / 30)
        + 0.5 * np.sin(lon * np.pi / 10)
    )


def continent_function(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Simplified continent-like pattern."""
    # Create rough continent shapes using Gaussians
    value = np.zeros_like(lon)

    # "Africa"
    value += 0.8 * np.exp(-((lon - 20) ** 2 + (lat - 5) ** 2) / 500)

    # "Eurasia"
    value += 0.9 * np.exp(-((lon - 80) ** 2 + (lat - 45) ** 2) / 1000)

    # "Americas"
    value += 0.85 * np.exp(-((lon + 80) ** 2 + (lat - 20) ** 2) / 800)

    return value


class SyntheticDataModule(pl.LightningDataModule):
    """Lightning DataModule for synthetic datasets."""

    def __init__(
        self,
        dataset_type: Literal["checkerboard", "gaussian_mixture", "sinusoidal"] = "checkerboard",
        n_samples: int = 10000,
        test_fraction: float = 0.3,
        batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        # Dataset-specific args
        grid_size: float = 10.0,
        n_centers: int = 10,
        sigma: float = 10.0,
        noise: float = 0.0,
        region: Optional[Tuple[float, float, float, float]] = None,
    ):
        """Initialize synthetic datamodule.

        Args:
            dataset_type: Type of synthetic dataset
            n_samples: Total samples
            test_fraction: Fraction for testing
            batch_size: Batch size
            num_workers: Dataloader workers
            seed: Random seed
            grid_size: Checkerboard grid size
            n_centers: Gaussian mixture centers
            sigma: Gaussian sigma
            noise: Noise level
            region: Bounding box
        """
        super().__init__()
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.grid_size = grid_size
        self.n_centers = n_centers
        self.sigma = sigma
        self.noise = noise
        self.region = region

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        if self.dataset_type == "checkerboard":
            full_dataset = CheckerboardDataset(
                n_samples=self.n_samples,
                grid_size=self.grid_size,
                noise=self.noise,
                region=self.region,
                seed=self.seed,
            )
        elif self.dataset_type == "gaussian_mixture":
            full_dataset = GaussianMixtureDataset(
                n_samples=self.n_samples,
                n_centers=self.n_centers,
                sigma=self.sigma,
                region=self.region,
                seed=self.seed,
            )
        elif self.dataset_type == "sinusoidal":
            full_dataset = FunctionFittingDataset(
                target_fn=sinusoidal_function,
                n_samples=self.n_samples,
                region=self.region,
                noise=self.noise,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        # Split
        n_total = len(full_dataset)
        n_test = int(n_total * self.test_fraction)
        n_train = n_total - n_test

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
