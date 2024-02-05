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
import rioxarray
import stackstac
from tqdm import tqdm


def set_up_parser():
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
        type="str",
        required=True,
        help="GeoParquet index file to sample from",
    )

    return parser


def main(args):
    assert not os.path.exists(args.output_fn)
    assert os.path.exists(args.s2_parquet_fn)
    
    container_client = azure.storage.blob.ContainerClient(
        args.storage_account,
        container_name=args.container_name,
        credential=args.sas_key,
    )
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("sentinel-2-l2a")

    df = pd.read_parquet(
        args.s2_parquet_fn
    )
    num_rows = df.shape[0]

    num_retries = 0
    num_error_hits = 0
    num_empty_hits = 0
    progress_bar = tqdm(total=args.high-args.low)
    results = []
    idx = args.low
    while idx < args.high:
        random_row = np.random.randint(0, num_rows)

        # attempt to get this item with progressive exponential backoff
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
            )
        _, num_channels, height, width = stack.shape

        x = np.random.randint(0, width - 256)
        y = np.random.randint(0, height - 256)

        patch = stack[0, :, y : y + 256, x : x + 256].compute()
        percent_empty = np.mean((np.isnan(patch.data)).sum(axis=0) == num_channels)

        if percent_empty > 0.1:
            num_empty_hits += 1
            continue
        else:
            with io.BytesIO() as buffer:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    patch = patch.astype(np.uint16)
                patch.rio.to_raster(
                    buffer, driver="COG", dtype=np.uint16, compress="LZW", predictor=2
                )
                buffer.seek(0)
                blob_client = container_client.get_blob_client(f"patch_{idx}.tif")
                blob_client.upload_blob(buffer, overwrite=True)
            results.append((idx, random_row, x, y))
            idx += 1
            progress_bar.update(1)
    progress_bar.close()

    df = pd.DataFrame(results, columns=["idx", "row", "x", "y"])
    df.to_csv(args.output_fn)

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