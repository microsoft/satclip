"""Smoke tests for the installed `satclip` package.

These verify that the package imports cleanly (i.e. that intra-package imports
resolve as a proper package) and that a small SatCLIP model can run a forward
pass end to end.
"""

import torch

import satclip
from satclip.model import SatCLIP


def test_public_api():
    for name in ("SatCLIP", "LocationEncoder", "SatCLIPLoss", "get_satclip"):
        assert hasattr(satclip, name), f"satclip.{name} should be importable"


def test_satclip_forward():
    model = SatCLIP(
        embed_dim=128,
        image_resolution=224,
        in_channels=13,
        vision_layers=2,
        vision_width=64,
        vision_patch_size=32,
        le_type="sphericalharmonics",
        pe_type="siren",
        legendre_polys=10,
        frequency_num=16,
        max_radius=360,
        min_radius=1,
        harmonics_calculation="analytic",
    )
    model.eval()

    img_batch = torch.randn(2, 13, 224, 224)
    loc_batch = torch.randn(2, 2)

    with torch.no_grad():
        logits_per_image, logits_per_coord = model(img_batch, loc_batch)

    assert logits_per_image.shape == (2, 2)
    assert logits_per_coord.shape == (2, 2)


def test_location_encoder_only():
    """The location encoder should embed coordinates on its own."""
    model = SatCLIP(
        embed_dim=128,
        image_resolution=224,
        in_channels=13,
        vision_layers=2,
        vision_width=64,
        vision_patch_size=32,
        le_type="sphericalharmonics",
        pe_type="siren",
        legendre_polys=10,
        frequency_num=16,
        max_radius=360,
        min_radius=1,
        harmonics_calculation="analytic",
    )
    model.eval()

    coords = torch.randn(4, 2)
    with torch.no_grad():
        emb = model.encode_location(coords)

    assert emb.shape[0] == 4
    assert emb.shape[1] == 128
