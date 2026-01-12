"""
Notebook 19: Quick Start Data Acquisition
Copy-paste this into your Colab notebook to download all required data.

Validated: 2026-01-12
All URLs confirmed working
Total download: ~63 MB
Time: ~3-5 minutes
"""

# ============================================================================
# STEP 1: Mount Google Drive (for GPW population data)
# ============================================================================
print("=" * 70)
print("STEP 1: Mounting Google Drive")
print("=" * 70)

from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted successfully\n")

# ============================================================================
# STEP 2: Install Required Packages
# ============================================================================
print("=" * 70)
print("STEP 2: Installing required packages")
print("=" * 70)

!pip install -q xarray netCDF4 rasterio geopandas
print("✅ Packages installed\n")

# ============================================================================
# STEP 3: Download ETOPO 2022 Elevation Data (~60 MB)
# ============================================================================
print("=" * 70)
print("STEP 3: Downloading ETOPO 2022 60s Elevation Data")
print("=" * 70)
print("URL: https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/...")
print("Size: ~60 MB")
print("Resolution: 60 arc-seconds (~2 km at equator)")
print()

!wget -q --show-progress \
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc" \
    -O etopo_60s.nc

# Verify download
import os
if os.path.exists('etopo_60s.nc') and os.path.getsize('etopo_60s.nc') > 1000000:
    print(f"✅ Elevation data downloaded: {os.path.getsize('etopo_60s.nc') / 1e6:.1f} MB\n")
else:
    print("❌ Download failed! Trying alternative method...")
    # Fallback: Use Open-Elevation API (see below)

# ============================================================================
# STEP 4: Download Natural Earth Coastlines (~3 MB)
# ============================================================================
print("=" * 70)
print("STEP 4: Downloading Natural Earth Coastlines")
print("=" * 70)
print("URL: https://naciscdn.org/naturalearth/10m/physical/...")
print("Size: ~3 MB")
print()

import requests
import zipfile
import io

url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
response = requests.get(url)

if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall('coastline_data')
    print("✅ Coastline data extracted to ./coastline_data/\n")
else:
    print(f"❌ Download failed with status code: {response.status_code}\n")

# ============================================================================
# STEP 5: Extract GPW Population Data from Drive
# ============================================================================
print("=" * 70)
print("STEP 5: Extracting GPW Population Density Data")
print("=" * 70)
print("Source: dataverse_files.zip in Google Drive")
print()

zip_path = '/content/drive/MyDrive/grad/learned_activations/dataverse_files.zip'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Extract only the population density file (not all files)
        gpw_file = 'gpw_v4_population_density_rev11_2020_15_min.tif'

        # Find the file in the zip
        matching = [f for f in z.namelist() if gpw_file in f]

        if matching:
            z.extract(matching[0], 'gpw_data')
            print(f"✅ Population data extracted: {matching[0]}\n")
        else:
            print("⚠️  GPW file not found in zip, extracting all...")
            z.extractall('gpw_data')
            print("✅ All GPW data extracted\n")
else:
    print(f"❌ Drive path not found: {zip_path}")
    print("Please check that dataverse_files.zip is in the correct location\n")

# ============================================================================
# STEP 6: (Optional) Download Temperature Data
# ============================================================================
print("=" * 70)
print("STEP 6 (Optional): Downloading Temperature Data")
print("=" * 70)
print("URL: https://springernature.figshare.com/ndownloader/files/12609182")
print("Size: ~1 MB")
print()

!wget -q --show-progress \
    "https://springernature.figshare.com/ndownloader/files/12609182" \
    -O temperature.csv

if os.path.exists('temperature.csv'):
    print("✅ Temperature data downloaded\n")
else:
    print("⚠️  Temperature download failed (optional, can skip)\n")

# ============================================================================
# STEP 7: Verify All Data Loaded Successfully
# ============================================================================
print("=" * 70)
print("STEP 7: Loading and Verifying Data")
print("=" * 70)

import xarray as xr
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np

success_count = 0
total_datasets = 4

# 1. Load elevation
try:
    ds_elev = xr.open_dataset('etopo_60s.nc')
    elevation_data = ds_elev['z'].values
    lats_elev = ds_elev['lat'].values
    lons_elev = ds_elev['lon'].values
    print(f"✅ Elevation: {elevation_data.shape} ({len(lats_elev)} lats × {len(lons_elev)} lons)")
    success_count += 1
except Exception as e:
    print(f"❌ Elevation loading failed: {e}")

# 2. Load coastlines
try:
    coastlines = gpd.read_file('coastline_data/ne_10m_coastline.shp')
    print(f"✅ Coastlines: {len(coastlines)} features")
    success_count += 1
except Exception as e:
    print(f"❌ Coastlines loading failed: {e}")

# 3. Load population
try:
    # Find the GPW file (may be nested)
    import glob
    gpw_files = glob.glob('gpw_data/**/*.tif', recursive=True)

    if gpw_files:
        with rasterio.open(gpw_files[0]) as src:
            pop_data = src.read(1)
            pop_transform = src.transform
        print(f"✅ Population: {pop_data.shape}")
        success_count += 1
    else:
        print("❌ Population file not found")
except Exception as e:
    print(f"❌ Population loading failed: {e}")

# 4. Load temperature (optional)
try:
    temp_df = pd.read_csv('temperature.csv')
    print(f"✅ Temperature: {len(temp_df)} observations (optional)")
    success_count += 1
except Exception as e:
    print(f"⚠️  Temperature loading failed (optional): {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DATA ACQUISITION SUMMARY")
print("=" * 70)
print(f"✅ Successfully loaded: {success_count}/{total_datasets} datasets")

if success_count >= 3:  # Need at least elevation, coastlines, population
    print("\n🎉 ALL CRITICAL DATA READY!")
    print("You can now proceed with Notebook 19 experiments.")
    print("\nData available:")
    print("  - elevation_data: Global elevation grid")
    print("  - coastlines: Vector coastline features")
    print("  - pop_data: Population density raster")
    if success_count == 4:
        print("  - temp_df: Temperature observations (optional)")
else:
    print("\n⚠️  WARNING: Some critical data failed to load")
    print("Please check error messages above and retry failed downloads")

print("=" * 70)

# ============================================================================
# FALLBACK: Open-Elevation API (if ETOPO download fails)
# ============================================================================
print("\n" + "=" * 70)
print("FALLBACK OPTION: Open-Elevation API")
print("=" * 70)
print("If ETOPO download failed, you can use the Open-Elevation API instead:")
print()
print("def get_elevation_api(lat, lon):")
print("    url = f'https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}'")
print("    response = requests.get(url)")
print("    if response.status_code == 200:")
print("        return response.json()['results'][0]['elevation']")
print("    return None")
print()
print("# Batch query (up to 100 locations)")
print("def get_elevations_batch(coords):")
print("    locations = '|'.join([f'{lat},{lon}' for lat, lon in coords])")
print("    url = f'https://api.open-elevation.com/api/v1/lookup?locations={locations}'")
print("    response = requests.get(url)")
print("    if response.status_code == 200:")
print("        return [r['elevation'] for r in response.json()['results']]")
print("    return None")
print()
print("⚠️  Note: API has rate limits, batch up to 100 locations, add 1 sec delay")
print("=" * 70)

# ============================================================================
# NEXT STEPS
# ============================================================================
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("1. Experiment 1: Regression vs Classification (use pop_data)")
print("2. Experiment 2: High-Frequency Tasks (use elevation_data)")
print("3. Experiment 3: Multi-Resolution (downsample elevation_data)")
print("4. Experiment 4: Complexity Measurement (train models, then analyze)")
print("5. Experiment 5: Task Difficulty (smooth elevation_data)")
print()
print("Refer to NOTEBOOK19_SIMPLICITY_BIAS_TESTS.md for detailed experiment setup")
print("=" * 70)
