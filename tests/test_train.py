"""End-to-end training smoke test.

Trains for a couple of steps on a tiny synthetic S2-100K-style dataset through
the same entry point as `satclip-train`, then reloads the checkpoint with
`get_satclip`. This exercises the packaged default config, the data module and
the checkpoint round trip.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch

import satclip
from satclip.load import get_satclip
from satclip.main import SatCLIPLightningModule, cli_main

N_TILES = 12
N_BANDS = 12  # S2-100K tiles omit B10; the training transform re-inserts it
TILE_SIZE = 256


@pytest.fixture
def s2_dataset(tmp_path):
    root = tmp_path / "s2"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    # The dataset integrity check expects the first and last S2-100K patch names.
    ids = [0, *range(1, N_TILES - 1), 99999]
    rows = []
    for i in ids:
        fn = f"patch_{i}.tif"
        with rasterio.open(
            root / "images" / fn, "w", driver="GTiff",
            height=TILE_SIZE, width=TILE_SIZE, count=N_BANDS, dtype="uint16",
        ) as f:
            f.write(rng.integers(0, 10000, (N_BANDS, TILE_SIZE, TILE_SIZE), dtype=np.uint16))
        rows.append({"fn": fn, "lon": rng.uniform(-180, 180), "lat": rng.uniform(-90, 90)})
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    return root


@pytest.mark.filterwarnings("ignore")
def test_train_and_reload(s2_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    package_configs = Path(satclip.__file__).parent / "configs"
    configs_before = sorted(p.name for p in package_configs.iterdir())

    cli_main(args=[
        f"--data.data_dir={s2_dataset}",
        "--data.batch_size=4",
        "--data.num_workers=0",
        "--trainer.accelerator=cpu",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=1",
        "--trainer.accumulate_grad_batches=1",
        "--trainer.enable_progress_bar=false",
    ])

    log_dir = tmp_path / "satclip_logs" / "satclip" / "satclip"
    assert (log_dir / "default-latest.yaml").is_file()
    # Training must not write into the installed package.
    assert sorted(p.name for p in package_configs.iterdir()) == configs_before

    ckpt_path = log_dir / "checkpoints" / "last.ckpt"
    assert ckpt_path.is_file()

    location_encoder = get_satclip(ckpt_path, device="cpu")
    with torch.no_grad():
        emb = location_encoder(torch.tensor([[-74.0, 40.7], [13.4, 52.5]]).double())
    assert emb.shape == (2, 256)
    assert torch.isfinite(emb).all()


def test_get_satclip_accepts_released_checkpoint_hparams(tmp_path):
    """Released checkpoints carry extra legacy hyperparameters that must be dropped."""
    module = SatCLIPLightningModule(
        embed_dim=32, image_resolution=64, vision_layers=2, vision_width=64,
        vision_patch_size=32, in_channels=13, legendre_polys=4,
    )
    ckpt = {
        "state_dict": module.state_dict(),
        "hyper_parameters": {
            **dict(module.hparams),
            "eval_downstream": False,
            "air_temp_data_path": None,
            "election_data_path": None,
        },
    }
    ckpt_path = tmp_path / "released.ckpt"
    torch.save(ckpt, ckpt_path)

    model = get_satclip(ckpt_path, device="cpu", return_all=True)
    assert isinstance(model, satclip.SatCLIP)
