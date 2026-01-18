#!/usr/bin/env python
"""Main training script for learned activations experiments.

Usage:
    # Using config files
    python -m experiments.train --config experiments/configs/experiments/elevation.yaml

    # Composing configs
    python -m experiments.train \\
        --config experiments/configs/base.yaml \\
        --config experiments/configs/experiments/elevation.yaml \\
        --config experiments/configs/encodings/sh_l10.yaml \\
        --config experiments/configs/activations/spline.yaml

    # With CLI overrides
    python -m experiments.train \\
        --config experiments/configs/experiments/elevation.yaml \\
        --model.hidden_dim=512 \\
        --training.learning_rate=0.0001

    # Quick test run
    python -m experiments.train --config ... --trainer.fast_dev_run=true
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.config import (
    load_config,
    merge_configs,
    parse_cli_overrides,
    Config,
)
from experiments.models.lightning_module import LearnedActivationsModule
from experiments.data import (
    GeographicDataModule,
    SyntheticDataModule,
    SatCLIPHuggingFaceDataModule,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train learned activations models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=[],
        help="Config file(s) to load. Can specify multiple for composition.",
    )

    # Allow arbitrary --key=value overrides
    args, unknown = parser.parse_known_args()

    # Parse CLI overrides
    cli_overrides = parse_cli_overrides(unknown)

    return args.config, cli_overrides


def get_datamodule(config: Config) -> pl.LightningDataModule:
    """Create datamodule from config.

    Args:
        config: Configuration object

    Returns:
        Lightning DataModule
    """
    data_config = config.data
    dataset = data_config.get("dataset", "checkerboard")

    if dataset in ["checkerboard", "gaussian_mixture", "sinusoidal"]:
        return SyntheticDataModule(
            dataset_type=dataset,
            n_samples=data_config.get("n_samples", 10000),
            test_fraction=data_config.get("test_fraction", 0.3),
            batch_size=data_config.get("batch_size", 256),
            num_workers=data_config.get("num_workers", 4),
            seed=config.experiment.get("seed", 42),
            grid_size=data_config.get("grid_size", 10.0),
            noise=data_config.get("noise", 0.0),
            region=data_config.get("region", None),
        )

    elif dataset in ["elevation", "population"]:
        return GeographicDataModule(
            task=dataset,
            n_samples=data_config.get("n_samples", 10000),
            region=data_config.get("region", None),
            test_fraction=data_config.get("test_fraction", 0.3),
            batch_size=data_config.get("batch_size", 256),
            num_workers=data_config.get("num_workers", 4),
            seed=config.experiment.get("seed", 42),
            data_path=data_config.get("data_path", None),
        )

    elif dataset == "satclip":
        return SatCLIPHuggingFaceDataModule(
            batch_size=data_config.get("batch_size", 512),
            num_workers=data_config.get("num_workers", 8),
            val_split=data_config.get("val_split", 0.1),
            cache_dir=data_config.get("cache_dir", None),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_model(config: Config, datamodule: pl.LightningDataModule) -> pl.LightningModule:
    """Create model from config.

    Args:
        config: Configuration object
        datamodule: DataModule (to get target statistics)

    Returns:
        Lightning Module
    """
    model_config = config.model
    training_config = config.training

    # Get target statistics if available
    target_mean = 0.0
    target_std = 1.0
    if hasattr(datamodule, "target_mean"):
        target_mean = float(datamodule.target_mean)
        target_std = float(datamodule.target_std)

    return LearnedActivationsModule(
        # Encoding
        encoding_type=model_config.encoding.get("type", "spherical_harmonics"),
        encoding_config={
            k: v for k, v in model_config.encoding.to_dict().items() if k != "type"
        },
        # Network
        hidden_dim=model_config.network.get("hidden_dim", 256),
        num_layers=model_config.network.get("num_layers", 3),
        # Activation
        activation_type=model_config.activation.get("type", "relu"),
        activation_config={
            k: v for k, v in model_config.activation.to_dict().items() if k != "type"
        },
        # Task
        task=config.data.get("task", "regression"),
        num_classes=config.data.get("num_classes", 1),
        # Training
        learning_rate=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0),
        # Normalization
        target_mean=target_mean,
        target_std=target_std,
    )


def get_callbacks(config: Config) -> List[pl.Callback]:
    """Create callbacks from config.

    Args:
        config: Configuration object

    Returns:
        List of callbacks
    """
    callbacks = []
    logging_config = config.logging
    training_config = config.training

    # Model checkpoint
    checkpoint_config = logging_config.get("checkpoint", Config({}))
    callbacks.append(
        ModelCheckpoint(
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_top_k=checkpoint_config.get("save_top_k", 1),
            save_last=checkpoint_config.get("save_last", True),
            filename="{epoch}-{val_loss:.4f}",
        )
    )

    # Early stopping
    early_stopping = training_config.get("early_stopping", Config({}))
    if early_stopping.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping.get("monitor", "val_loss"),
                mode=early_stopping.get("mode", "min"),
                patience=early_stopping.get("patience", 10),
            )
        )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    return callbacks


def get_logger(config: Config) -> pl.loggers.Logger:
    """Create logger from config.

    Args:
        config: Configuration object

    Returns:
        Logger instance
    """
    logging_config = config.logging
    logger_type = logging_config.get("logger", "tensorboard")

    experiment_name = config.experiment.get("name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=logging_config.get("save_dir", "./logs"),
            name=logging_config.get("name", "experiments"),
            version=f"{experiment_name}_{timestamp}",
        )
    elif logger_type == "csv":
        return CSVLogger(
            save_dir=logging_config.get("save_dir", "./logs"),
            name=f"{experiment_name}_{timestamp}",
        )
    else:
        # Default to TensorBoard
        return TensorBoardLogger(
            save_dir="./logs",
            name=experiment_name,
        )


def main():
    """Main training function."""
    # Parse arguments
    config_paths, cli_overrides = parse_args()

    # Load base config
    base_config_path = Path(__file__).parent / "configs" / "base.yaml"
    configs = [load_config(base_config_path)]

    # Load specified configs
    for path in config_paths:
        configs.append(load_config(path))

    # Merge all configs
    merged = merge_configs(configs)

    # Apply CLI overrides
    merged = merge_configs([merged, cli_overrides])

    # Convert to Config object
    config = Config(merged)

    # Print config
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Experiment: {config.experiment.get('name', 'unknown')}")
    print(f"Encoding: {config.model.encoding.get('type', 'unknown')}")
    print(f"Activation: {config.model.activation.get('type', 'unknown')}")
    print(f"Dataset: {config.data.get('dataset', 'unknown')}")
    print("=" * 60)

    # Set seed
    seed = config.experiment.get("seed", 42)
    pl.seed_everything(seed)

    # Create datamodule
    print("\nSetting up data...")
    datamodule = get_datamodule(config)
    datamodule.setup()

    # Create model
    print("Creating model...")
    model = get_model(config, datamodule)

    # Create callbacks and logger
    callbacks = get_callbacks(config)
    logger = get_logger(config)

    # Create trainer
    hardware = config.get("hardware", Config({}))
    training = config.training

    trainer = pl.Trainer(
        max_epochs=training.get("max_epochs", 100),
        accelerator=hardware.get("accelerator", "auto"),
        devices=hardware.get("devices", 1),
        precision=hardware.get("precision", 32),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=training.get("gradient_clip_val", None),
        accumulate_grad_batches=training.get("accumulate_grad_batches", 1),
        enable_progress_bar=True,
        deterministic=config.get("reproducibility", Config({})).get("deterministic", True),
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
