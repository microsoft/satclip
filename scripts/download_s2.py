import argparse
import io
import os
import time
import warnings

import azure.storage.blob
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray  # rioxarray is required for the .rio methods in xarray despite what mypy, ruff, etc. says :)
import stackstac
from tqdm import tqdm


def set_up_parser() -> argparse.ArgumentParser:
    """
    Set up and return a command-line argument parser for the Sentinel-2 patch downloader.

    The parser defines required and optional arguments for specifying the patch download range,
    Azure blob storage configuration, and the source GeoParquet file used to sample Sentinel-2 items.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all necessary CLI options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--low",
        default=0,
        type=int,
        required=True,
        help="Starting index",
    )

    parser.add_argument(
        "--high",
        default=100_000,
        type=int,
        required=True,
        help="Ending index",
    )

    parser.add_argument(
        "--output_fn",
        default="patch_locations.csv",
        type=str,
        required=True,
        help="Output filename",
    )

    parser.add_argument(
        "--storage_account",
        type=str,
        required=True,
        help="Azure storage account URL (e.g. 'https://storageaccount.blob.core.windows.net')",
    )

    parser.add_argument(
        "--container_name",
        type=str,
        required=True,
        help="Azure blob container name",
    )

    parser.add_argument(
        "--sas_key",
        type=str,
        required=True,
        help="SAS key for Azure blob container",
    )

    parser.add_argument(
        "--s2_parquet_fn",
        type=str,
        required=True,
        help="GeoParquet index file to sample from",
    )

    return parser


def main(args):
    """
    Main processing function for downloading Sentinel-2 image patches from a STAC catalog
    and uploading them to an Azure Blob container as Cloud-Optimized GeoTIFFs (COGs).

    The function selects valid image patches from the input GeoParquet file,
    extracts a 256x256 region from a Sentinel-2 STAC item, filters based on NaN content,
    and embeds relevant metadata before uploading the patch to Azure Blob Storage.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, including input range, output file, Azure credentials,
        and STAC sampling source.
    """
    # Sanity checks: output file shouldn't already exist, input parquet must exist
    assert not os.path.exists(args.output_fn)
    assert os.path.exists(args.s2_parquet_fn)

    # Set up Azure blob container client
    container_client = azure.storage.blob.ContainerClient(
        args.storage_account,
        container_name=args.container_name,
        credential=args.sas_key,
    )

    # Connect to Microsoft Planetary Computer STAC API
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace,
    )

    collection = catalog.get_collection("sentinel-2-l2a")

    # Load input patch candidates from parquet
    df = pd.read_parquet(args.s2_parquet_fn)
    num_rows = df.shape[0]

    # Initialize stats and result tracking
    num_retries = 0
    num_error_hits = 0
    num_empty_hits = 0
    progress_bar = tqdm(total=args.high - args.low)
    results = []

    # Begin sampling loop
    idx = args.low
    while idx < args.high:
        # Select a random row from GeoParquet file
        random_row = np.random.randint(0, num_rows)

        # Attempt to get this item with progressive exponential backoff
        item = None
        for j in range(4):
            try:
                item = collection.get_item(df.iloc[random_row]["id"])
                break
            except Exception as e:
                print(e)
                print("retrying", random_row, j)
                num_retries += 1
                time.sleep(2**j)

        if item is None:
            print(f"failed to get item {random_row}")
            num_error_hits += 1
            continue

        # Load selected STAC item into a multi-band raster stack
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack = stackstac.stack(
                item,
                assets=[
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B11",
                    "B12",
                ],
                epsg=4326,
            )
        _, num_channels, height, width = stack.shape

        # Randomly sample a 256x256 window within image bounds
        x = np.random.randint(0, width - 256)
        y = np.random.randint(0, height - 256)

        # Extract patch and compute in-memory
        patch = stack[0, :, y : y + 256, x : x + 256].compute()

        # Filter patches with more than 10% missing data
        percent_empty = np.mean((np.isnan(patch.data)).sum(axis=0) == num_channels)
        percent_zero = np.mean((patch.data == 0).sum(axis=0) == num_channels)

        if percent_empty > 0.1 or percent_zero > 0.1:
            num_empty_hits += 1
            continue

        # Save valid patch to Azure Blob Storage as GeoTIFF with metadata
        with io.BytesIO() as buffer:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                patch = patch.astype(np.uint16)

            # Extract STAC metadata for provenance and traceability
            metadata = {
                "datetime": item.datetime.isoformat(),
                "platform": item.properties.get("platform", ""),
                "mgrs_tile": item.properties.get("s2:mgrs_tile", ""),
                "granule_id": item.properties.get("s2:granule_id", ""),
                "orbit_state": item.properties.get("sat:orbit_state", ""),
                "relative_orbit": str(item.properties.get("sat:relative_orbit", "")),
                "cloud_cover": str(item.properties.get("eo:cloud_cover", "")),
                "mean_solar_zenith": str(
                    item.properties.get("s2:mean_solar_zenith", "")
                ),
                "mean_solar_azimuth": str(
                    item.properties.get("s2:mean_solar_azimuth", "")
                ),
            }

            # Attach metadata to the patch for inclusion in the raster tags
            patch.attrs.update(metadata)

            # Write Cloud-Optimized GeoTIFF to memory
            patch.rio.to_raster(
                buffer,
                driver="GTiff",
                dtype=np.uint16,
                compress="LZW",
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                interleave="pixel",
            )

            # Upload patch to Azure Blob
            buffer.seek(0)
            blob_client = container_client.get_blob_client(f"patch_{idx}.tif")
            blob_client.upload_blob(buffer, overwrite=True)

            # Store patch info for CSV log
            results.append(
                (
                    idx,
                    random_row,
                    x,
                    y,
                    metadata["granule_id"],
                )
            )

            idx += 1
            progress_bar.update(1)

    progress_bar.close()

    # Save all patch locations and sample info to CSV
    df = pd.DataFrame(results, columns=["idx", "row", "x", "y", "granule_id"])
    df.to_csv(args.output_fn)

    # Print final stats
    print("Summary:")
    print(f"range: [{args.low}, {args.high})")
    print(f"num hits: {len(results)}")
    print(f"num empty hits: {num_empty_hits}")
    print(f"num error hits: {num_error_hits}")
    print(f"num retries: {num_retries}")


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
