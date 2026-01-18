"""HuggingFace dataset loader for SatCLIP data.

Uses the dataset: davanstrien/satclip
https://huggingface.co/datasets/davanstrien/satclip
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from typing import Optional, Dict, Any, Callable
import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class SatCLIPHuggingFaceDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace SatCLIP data.

    The HuggingFace dataset contains:
    - image: Satellite imagery
    - lon: Longitude coordinate
    - lat: Latitude coordinate
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ):
        """Initialize dataset.

        Args:
            split: Dataset split ("train", "test")
            transform: Optional transform to apply to images
            cache_dir: Directory to cache downloaded data
            streaming: Whether to stream data (saves memory)
        """
        if not HAS_DATASETS:
            raise ImportError(
                "Please install datasets: pip install datasets"
            )

        self.transform = transform
        self.streaming = streaming

        # Load from HuggingFace
        self.dataset = load_dataset(
            "davanstrien/satclip",
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        if not streaming:
            self._length = len(self.dataset)

    def __len__(self) -> int:
        if self.streaming:
            raise NotImplementedError("Length not available in streaming mode")
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Dict with 'image', 'point' keys
        """
        sample = self.dataset[idx]

        # Extract coordinates
        lon = sample["lon"]
        lat = sample["lat"]
        point = torch.tensor([lon, lat], dtype=torch.float32)

        # Process image
        image = sample["image"]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        elif hasattr(image, "numpy"):
            image = torch.from_numpy(np.array(image)).float()

        # Ensure correct shape (C, H, W)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[-1] in [1, 3, 4, 13]:
            # HWC -> CHW
            image = image.permute(2, 0, 1)

        # Apply transform
        if self.transform is not None:
            sample_dict = {"image": image, "point": point}
            sample_dict = self.transform(sample_dict)
            return sample_dict

        return {"image": image, "point": point}


class SatCLIPHuggingFaceDataModule(pl.LightningDataModule):
    """Lightning DataModule for HuggingFace SatCLIP dataset.

    Provides train/val/test dataloaders for contrastive learning.
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        pin_memory: bool = True,
    ):
        """Initialize datamodule.

        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            val_split: Fraction of training data for validation
            cache_dir: Directory to cache downloaded data
            transform: Optional transform for images
            pin_memory: Whether to pin memory (faster GPU transfer)
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.cache_dir = cache_dir
        self.transform = transform
        self.pin_memory = pin_memory

        self.save_hyperparameters(ignore=["transform"])

    def prepare_data(self):
        """Download data if needed."""
        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        # This downloads the dataset
        load_dataset("davanstrien/satclip", cache_dir=self.cache_dir)

    def setup(self, stage: Optional[str] = None):
        """Set up train/val/test datasets."""
        full_dataset = SatCLIPHuggingFaceDataset(
            split="train",
            transform=self.transform,
            cache_dir=self.cache_dir,
        )

        # Split into train/val
        n_total = len(full_dataset)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_val

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class PointsOnlyDataset(Dataset):
    """Dataset that returns only coordinates (no images).

    Useful for training location encoders without images.
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ):
        """Initialize dataset.

        Args:
            split: Dataset split
            cache_dir: Cache directory
        """
        if not HAS_DATASETS:
            raise ImportError("Please install datasets: pip install datasets")

        self.dataset = load_dataset(
            "davanstrien/satclip",
            split=split,
            cache_dir=cache_dir,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get coordinates only.

        Args:
            idx: Sample index

        Returns:
            Coordinate tensor of shape (2,)
        """
        sample = self.dataset[idx]
        return torch.tensor([sample["lon"], sample["lat"]], dtype=torch.float32)
