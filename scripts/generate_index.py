from azure.storage.blob import ContainerClient
import rasterio
import fiona
from tqdm import tqdm
import fiona.transform
import pandas as pd
import shapely.geometry


ACCOUNT_URL = ""
CONTAINER_NAME = ""
SAS_KEY = ""


def main():
    client = ContainerClient(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER_NAME,
        credential=SAS_KEY
    )
    prefix = "satclip-1m/"
    blobs = client.list_blobs(name_starts_with=prefix)


    urls = []
    for blob in tqdm(blobs):
        urls.append(f"{ACCOUNT_URL}/{CONTAINER_NAME}/{blob.name}{SAS_KEY}")
    non_sas_urls = [
        url.split("?")[0] for url in urls
    ]


    lats = []
    lons = []
    for url in tqdm(urls):
        with rasterio.open(url) as src:
            geom = shapely.geometry.mapping(shapely.geometry.box(*src.bounds))
            warped_geom = fiona.transform.transform_geom(src.crs, "EPSG:4326", geom)
            shape = shapely.geometry.shape(warped_geom)
            x, y = shape.centroid.xy
            x = x[0]
            y = y[0]
            lats.append(y)
            lons.append(x)


    df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "url": non_sas_urls,
    })
    df.to_csv("satclip-300k_index.csv", index=False)


if __name__ == '__main__':
    main()
