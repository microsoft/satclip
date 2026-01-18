"""Geographic datasets for regression experiments.

Includes:
- Elevation data (ETOPO)
- Population density (GPW)
- Custom geographic data loading
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from typing import Optional, Tuple, Literal
import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


class GeoDataset(Dataset):
    """Base dataset for geographic (coordinate, value) pairs."""

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        normalize_values: bool = True,
    ):
        """Initialize dataset.

        Args:
            coords: (N, 2) array of (lon, lat) coordinates
            values: (N,) array of target values
            normalize_values: Whether to normalize target values
        """
        self.coords = torch.from_numpy(coords).float()
        self.values = torch.from_numpy(values).float()

        self.normalize_values = normalize_values
        if normalize_values:
            self.mean = self.values.mean()
            self.std = self.values.std()
        else:
            self.mean = 0.0
            self.std = 1.0

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict:
        """Get sample.

        Args:
            idx: Sample index

        Returns:
            Dict with 'coords' and 'target' keys
        """
        return {
            "coords": self.coords[idx],
            "target": self.values[idx],
        }


class ElevationDataset(GeoDataset):
    """Dataset for elevation prediction task.

    Downloads ETOPO elevation data if not available.
    """

    ETOPO_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"

    def __init__(
        self,
        n_samples: int = 10000,
        region: Optional[Tuple[float, float, float, float]] = None,
        seed: int = 42,
        data_path: Optional[str] = None,
    ):
        """Initialize elevation dataset.

        Args:
            n_samples: Number of samples to use
            region: Optional (lon_min, lon_max, lat_min, lat_max) bounding box
            seed: Random seed for sampling
            data_path: Path to pre-downloaded ETOPO data
        """
        if not HAS_XARRAY:
            raise ImportError("Please install xarray: pip install xarray netCDF4")

        coords, values = self._load_elevation_data(
            n_samples=n_samples,
            region=region,
            seed=seed,
            data_path=data_path,
        )

        super().__init__(coords, values)

    def _load_elevation_data(
        self,
        n_samples: int,
        region: Optional[Tuple],
        seed: int,
        data_path: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and sample elevation data.

        Args:
            n_samples: Number of samples
            region: Optional bounding box
            seed: Random seed
            data_path: Data file path

        Returns:
            Tuple of (coords, values) arrays
        """
        import urllib.request
        import os

        # Download if needed
        if data_path is None:
            data_path = "etopo_60s.nc"

        if not os.path.exists(data_path):
            print(f"Downloading ETOPO data to {data_path}...")
            urllib.request.urlretrieve(self.ETOPO_URL, data_path)

        # Load data
        ds = xr.open_dataset(data_path)
        elevation = ds["z"].values
        lats = ds["lat"].values
        lons = ds["lon"].values

        # Create coordinate grids
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        elev_flat = elevation.flatten()

        # Remove NaN values
        valid_mask = ~np.isnan(elev_flat)
        lon_flat = lon_flat[valid_mask]
        lat_flat = lat_flat[valid_mask]
        elev_flat = elev_flat[valid_mask]

        # Apply region filter
        if region is not None:
            lon_min, lon_max, lat_min, lat_max = region
            region_mask = (
                (lon_flat >= lon_min)
                & (lon_flat <= lon_max)
                & (lat_flat >= lat_min)
                & (lat_flat <= lat_max)
            )
            lon_flat = lon_flat[region_mask]
            lat_flat = lat_flat[region_mask]
            elev_flat = elev_flat[region_mask]

        # Random sample
        np.random.seed(seed)
        n_available = len(lon_flat)
        sample_size = min(n_samples, n_available)
        indices = np.random.choice(n_available, size=sample_size, replace=False)

        coords = np.column_stack([lon_flat[indices], lat_flat[indices]])
        values = elev_flat[indices]

        return coords, values


class PopulationDataset(GeoDataset):
    """Dataset for population density prediction.

    Uses GPW (Gridded Population of the World) data.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        region: Optional[Tuple[float, float, float, float]] = None,
        resolution: Literal["15_min", "30_min", "1_degree"] = "15_min",
        log_transform: bool = True,
        seed: int = 42,
        data_path: Optional[str] = None,
    ):
        """Initialize population dataset.

        Args:
            n_samples: Number of samples
            region: Optional bounding box
            resolution: Data resolution ("15_min", "30_min", "1_degree")
            log_transform: Whether to log-transform population values
            seed: Random seed
            data_path: Path to GPW data file
        """
        self.log_transform = log_transform

        coords, values = self._load_population_data(
            n_samples=n_samples,
            region=region,
            resolution=resolution,
            seed=seed,
            data_path=data_path,
        )

        super().__init__(coords, values)

    def _load_population_data(
        self,
        n_samples: int,
        region: Optional[Tuple],
        resolution: str,
        seed: int,
        data_path: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load population data.

        Note: GPW data requires manual download from SEDAC.
        This method expects rasterio-readable files.
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("Please install rasterio: pip install rasterio")

        if data_path is None:
            raise ValueError(
                "Population data requires manual download from SEDAC. "
                "Please provide data_path to the GPW GeoTIFF file."
            )

        with rasterio.open(data_path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            height, width = data.shape

            # Get coordinates
            lons = np.array([transform * (i, 0) for i in range(width)])[:, 0]
            lats = np.array([transform * (0, j) for j in range(height)])[:, 1]

            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Handle nodata
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        pop_flat = data.flatten()

        # Valid data only
        valid_mask = (pop_flat > 0) & ~np.isnan(pop_flat)
        lon_flat = lon_flat[valid_mask]
        lat_flat = lat_flat[valid_mask]
        pop_flat = pop_flat[valid_mask]

        # Apply region filter
        if region is not None:
            lon_min, lon_max, lat_min, lat_max = region
            region_mask = (
                (lon_flat >= lon_min)
                & (lon_flat <= lon_max)
                & (lat_flat >= lat_min)
                & (lat_flat <= lat_max)
            )
            lon_flat = lon_flat[region_mask]
            lat_flat = lat_flat[region_mask]
            pop_flat = pop_flat[region_mask]

        # Log transform
        if self.log_transform:
            pop_flat = np.log1p(pop_flat)

        # Random sample
        np.random.seed(seed)
        n_available = len(lon_flat)
        sample_size = min(n_samples, n_available)
        indices = np.random.choice(n_available, size=sample_size, replace=False)

        coords = np.column_stack([lon_flat[indices], lat_flat[indices]])
        values = pop_flat[indices]

        return coords, values


class GeographicDataModule(pl.LightningDataModule):
    """Lightning DataModule for geographic regression tasks."""

    def __init__(
        self,
        task: Literal["elevation", "population"] = "elevation",
        n_samples: int = 10000,
        region: Optional[Tuple[float, float, float, float]] = None,
        test_fraction: float = 0.3,
        batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        data_path: Optional[str] = None,
    ):
        """Initialize datamodule.

        Args:
            task: Task type ("elevation" or "population")
            n_samples: Total number of samples
            region: Optional bounding box
            test_fraction: Fraction of data for testing
            batch_size: Batch size
            num_workers: Dataloader workers
            seed: Random seed
            data_path: Path to data file
        """
        super().__init__()
        self.task = task
        self.n_samples = n_samples
        self.region = region
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.data_path = data_path

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        if self.task == "elevation":
            full_dataset = ElevationDataset(
                n_samples=self.n_samples,
                region=self.region,
                seed=self.seed,
                data_path=self.data_path,
            )
        elif self.task == "population":
            full_dataset = PopulationDataset(
                n_samples=self.n_samples,
                region=self.region,
                seed=self.seed,
                data_path=self.data_path,
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Store normalization parameters
        self.target_mean = full_dataset.mean
        self.target_std = full_dataset.std

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

    def test_dataloader(self):
        return self.val_dataloader()
