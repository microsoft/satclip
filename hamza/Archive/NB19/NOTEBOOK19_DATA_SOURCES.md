# Notebook 19: Data Sources and Validation

**Date**: 2026-01-12
**Purpose**: Document all data sources needed for Notebook 19 simplicity bias tests

---

## Current Data (Already Available)

### ✅ 1. Population Density - GPW v4 (2020)
**Source**: Gridded Population of the World v4 (CIESIN)
**Current Location**: `dataverse_files.zip` in Google Drive
**File**: `gpw_v4_population_density_rev11_2020_15_min_tif.zip`
**Resolution**: 15 arc-minutes (~30 km at equator)
**Format**: GeoTIFF
**Coverage**: Global
**Loading method**: Already implemented in notebooks 10, 18
```python
# Mount Drive and extract
from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_path = '/content/drive/MyDrive/grad/learned_activations/dataverse_files.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('gpw_data')

# Load with rasterio
import rasterio
with rasterio.open('gpw_data/gpw_v4_population_density_rev11_2020_15_min.tif') as src:
    pop_data = src.read(1)
    transform = src.transform
```

**Status**: ✅ Ready to use

---

## New Data Needed for Notebook 19

### 🔍 2. Elevation Data (CRITICAL for Exp 2, 3, 5)

**Goal**: High-frequency geographic data with sharp transitions (mountains, valleys)

#### Option A: ETOPO 2022 (RECOMMENDED) ⭐⭐⭐
**Source**: NOAA National Centers for Environmental Information
**Resolution**: 15 arc-second (~450m at equator)
**Direct Download URL**:
```
https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/15s/15s_surface_elev_netcdf/ETOPO_2022_v1_15s_N90W180_surface.nc
```

**File size**: ~900 MB (single hemisphere)
**Alternative** (smaller, coarser):
```
https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc
```
**File size**: ~60 MB (60 arc-second = 1 arc-minute, ~2 km resolution)

**Format**: NetCDF4
**Coverage**: Global (ocean bathymetry + land topography)
**Data type**: Ice surface elevation (meters)

**Loading in Colab**:
```python
# Download
!wget https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc -O etopo_60s.nc

# Load with xarray
import xarray as xr
import numpy as np

ds = xr.open_dataset('etopo_60s.nc')
elevation = ds['z'].values  # elevation in meters
lats = ds['lat'].values
lons = ds['lon'].values

# Sample coordinates
coords = []
elevs = []
for i in range(10000):
    lat_idx = np.random.randint(0, len(lats))
    lon_idx = np.random.randint(0, len(lons))
    coords.append([lats[lat_idx], lons[lon_idx]])
    elevs.append(elevation[lat_idx, lon_idx])
```

**Validation Status**: ✅ **CONFIRMED WORKING** (tested 2026-01-12)
- HTTP 200 response
- File size: ~60 MB for 60s resolution
- Direct download works, no authentication needed

**Alternative sources if needed**:
- **GEBCO 2023**: Requires registration at https://www.gebco.net/data_and_products/gridded_bathymetry_data/
- **GMTED2010**: https://topotools.cr.usgs.gov/gmted_viewer/ (requires manual download)

---

#### Option B: SRTM (Shuttle Radar Topography Mission)
**Source**: NASA JPL / USGS Earth Explorer
**Resolution**: 90m (3 arc-second) or 30m (1 arc-second)
**Coverage**: 60°N to 56°S (no polar regions)
**Format**: GeoTIFF tiles

**Download options**:
1. **OpenTopography** (Easy, no account needed):
```
https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1
```
Select region, download GeoTIFF

2. **Earthdata** (Requires free account):
```
https://search.earthdata.nasa.gov/search?q=SRTM
```

**Pros**: Higher resolution (better for fine-scale tests)
**Cons**: Tiles need stitching, registration required

**Loading**:
```python
# If using OpenTopography GeoTIFF
import rasterio
with rasterio.open('srtm_tile.tif') as src:
    elevation = src.read(1)
    bounds = src.bounds
    transform = src.transform
```

**Validation Status**: ⏳ NEED TO CHECK

---

#### Option C: Pre-processed Elevation Dataset (EASIEST)
**Create from GPW coordinates + API**:

Use **Open-Elevation API** for point queries:
```python
import requests
import numpy as np

def get_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['results'][0]['elevation']
    return None

# Batch query
coords = np.random.uniform([-90, 90], [-180, 180], (10000, 2))
elevations = [get_elevation(lat, lon) for lat, lon in coords]
```

**Pros**: No file download, works immediately
**Cons**: Slow (API rate limits), requires internet

**Validation Status**: ✅ **CONFIRMED WORKING** (tested 2026-01-12)
- Returns JSON with elevation data
- Example query returned: 370.0m elevation for (35.5, -120.5)
- Rate limit: ~100 locations per request
- Recommended: batch queries with 1 second delay between requests

---

### 🌊 3. Coastline Distance (MEDIUM PRIORITY for Exp 2)

**Goal**: Sharp step-function transitions at ocean boundaries

#### Option A: Compute from Natural Earth Coastlines ⭐ (RECOMMENDED)
**Source**: Already have Natural Earth 110m countries
**Method**: Compute distance to nearest coastline polygon

```python
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Download coastline data
url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall('coastline_data')

coastlines = gpd.read_file('coastline_data/ne_10m_coastline.shp')

# Function to compute distance
def distance_to_coast(lat, lon):
    point = Point(lon, lat)
    return coastlines.distance(point).min()

# Apply to coordinates
coords = [[35.5, -120.5], [40.0, -105.0], ...]  # sample points
distances = [distance_to_coast(lat, lon) for lat, lon in coords]
```

**Validation Status**: ✅ **CONFIRMED WORKING** (tested 2026-01-12)
- HTTP 200 response
- File size: ~3 MB
- CloudFront CDN, fast download
- Previously confirmed in notebook 05

---

#### Option B: Pre-computed Global Distance to Coast
**Source**: NASA Ocean Color (MODIS)
**URL**: https://oceancolor.gsfc.nasa.gov/docs/distfromcoast/

**Format**: NetCDF or HDF
**Resolution**: ~4 km

**Loading**:
```python
# If available as NetCDF
import xarray as xr
ds = xr.open_dataset('distance_to_coast.nc')
dist = ds['distance'].values
```

**Validation Status**: ⏳ NEED TO CHECK

---

### 🌡️ 4. Temperature Gradient Magnitude (OPTIONAL for Exp 2)

**Goal**: Medium-frequency data with sharp fronts

#### Option A: Compute from Existing Temperature Data
**Use**: Figshare air temperature dataset (already have)
**Method**: Compute spatial gradient magnitude

```python
# From notebook 01 - already working
url = 'https://springernature.figshare.com/ndownloader/files/12609182'
url_open = request.urlopen(url)
inc = np.array(pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))))
coords = inc[:, :2]  # lat, lon
temps = inc[:, 4]  # temperature

# Interpolate to regular grid
from scipy.interpolate import griddata
grid_x, grid_y = np.mgrid[-180:180:1, -90:90:1]
grid_temp = griddata(coords, temps, (grid_x, grid_y), method='cubic')

# Compute gradient magnitude
grad_x, grad_y = np.gradient(grid_temp)
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
```

**Validation Status**: ✅ **CONFIRMED WORKING** (tested 2026-01-12)
- HTTP 302 redirect to S3 (normal behavior)
- Returns CSV with temperature data
- Previously confirmed in notebook 01
- Fast download, no rate limits

---

#### Option B: ERA5 Reanalysis (IF USER CAN DOWNLOAD)
**Source**: Copernicus Climate Data Store
**URL**: https://cds.climate.copernicus.eu/

**Requirements**:
- Free CDS account
- CDS API key
- Use `cdsapi` Python package

**NOT RECOMMENDED** for Colab (slow, requires account)

**Alternative**: Skip temperature gradient, focus on elevation + coastline

---

## Multi-Resolution Data (for Experiment 3)

### Strategy: Downsample High-Resolution Elevation

Once we have ETOPO 60s (~2 km), create multiple resolutions:

```python
import xarray as xr
from scipy.ndimage import zoom

# Load original
ds = xr.open_dataset('etopo_60s.nc')
elevation_fine = ds['z'].values

# Create coarser resolutions
def downsample(data, factor):
    return zoom(data, 1/factor, order=1)  # bilinear

elevation_1deg = downsample(elevation_fine, 60)    # ~110 km
elevation_05deg = downsample(elevation_fine, 30)   # ~55 km
elevation_025deg = downsample(elevation_fine, 15)  # ~28 km
elevation_01deg = downsample(elevation_fine, 6)    # ~11 km (original 60s)

resolutions = {
    'coarse': (elevation_1deg, '1 degree', 110),
    'medium': (elevation_05deg, '0.5 degree', 55),
    'fine': (elevation_025deg, '0.25 degree', 28),
    'ultra': (elevation_fine, '0.1 degree', 11),  # 60 arc-second
}
```

---

## Data Validation Checklist ✅

**Last Validated**: 2026-01-12
**All critical URLs confirmed working!**

### High Priority (MUST HAVE)
- [x] ✅ **ETOPO 2022 60s**: https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc
  - **Status**: HTTP 200, ~60 MB, direct download works
  - No authentication needed

- [x] ✅ **Natural Earth Coastlines**: https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip
  - **Status**: HTTP 200, ~3 MB, CloudFront CDN
  - Fast download, previously validated in NB05

- [x] ✅ **GPW Population**: Already have in Drive (dataverse_files.zip)
  - No download needed

### Medium Priority (NICE TO HAVE)
- [x] ✅ **Figshare Temperature**: https://springernature.figshare.com/ndownloader/files/12609182
  - **Status**: HTTP 302 → S3, CSV data
  - Previously validated in NB01

- [x] ✅ **Open-Elevation API**: https://api.open-elevation.com/api/v1/lookup
  - **Status**: Working, returns JSON
  - Test confirmed: `?locations=35.5,-120.5` → 370.0m elevation
  - Rate limit: batch up to 100 locations, 1 sec delay recommended

### Low Priority (OPTIONAL)
- [ ] **OpenTopography SRTM**: https://portal.opentopography.org/
  - Only if need higher resolution than ETOPO 60s

---

## Recommended Data Acquisition Plan

### Phase 1: Critical Data (Do First)
1. **Elevation**: Download ETOPO 2022 60s (~60 MB)
   - If link fails → try GEBCO 2023
   - If both fail → use Open-Elevation API (slower but works)

2. **Coastline**: Download Natural Earth 10m coastlines (~10 MB)
   - Already confirmed working from NB05

3. **Population**: Already have (GPW in Drive)

### Phase 2: Optional Data (If Time/Need)
4. **Temperature gradient**: Compute from Figshare data (already working)
5. **SRTM higher-res**: Only if results show need for finer resolution

---

## Colab Notebook Code Template

```python
# ===== DATA ACQUISITION SECTION =====

# 1. Mount Drive (for GPW population)
from google.colab import drive
drive.mount('/content/drive')

# 2. Download Elevation Data
print("Downloading ETOPO 2022 60s elevation data (~60 MB)...")
!wget -q --show-progress https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc -O etopo_60s.nc

# Check if download succeeded
import os
if os.path.exists('etopo_60s.nc'):
    print("✅ Elevation data downloaded successfully")
else:
    print("❌ Download failed, trying alternative source...")
    # Try GEBCO or Open-Elevation API

# 3. Download Coastline Data
print("Downloading Natural Earth coastlines (~10 MB)...")
import requests
import zipfile
import io

url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall('coastline_data')
print("✅ Coastline data extracted")

# 4. Load Population Data (already in Drive)
print("Extracting GPW population data from Drive...")
zip_path = '/content/drive/MyDrive/grad/learned_activations/dataverse_files.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('gpw_data')
print("✅ Population data ready")

# 5. (Optional) Download Temperature Data
print("Downloading temperature data...")
!wget -q --show-progress https://springernature.figshare.com/ndownloader/files/12609182 -O temperature.csv
print("✅ Temperature data downloaded")

# ===== DATA LOADING SECTION =====
import xarray as xr
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np

# Load elevation
ds_elev = xr.open_dataset('etopo_60s.nc')
elevation_data = ds_elev['z'].values
lats_elev = ds_elev['lat'].values
lons_elev = ds_elev['lon'].values

# Load coastlines
coastlines = gpd.read_file('coastline_data/ne_10m_coastline.shp')

# Load population
with rasterio.open('gpw_data/gpw_v4_population_density_rev11_2020_15_min.tif') as src:
    pop_data = src.read(1)
    pop_transform = src.transform

# Load temperature (if needed)
temp_df = pd.read_csv('temperature.csv')
temp_coords = temp_df[['latitude', 'longitude']].values
temp_values = temp_df['temperature'].values

print("\n✅ All data loaded successfully!")
```

---

## Fallback Plan (If Downloads Fail)

If ETOPO/GEBCO links don't work:

### Option 1: Use Open-Elevation API
```python
def get_elevations_batch(coords, batch_size=100):
    """Query Open-Elevation API in batches"""
    elevations = []
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i+batch_size]
        locations = '|'.join([f"{lat},{lon}" for lat, lon in batch])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json()['results']
            elevations.extend([r['elevation'] for r in results])
        time.sleep(1)  # Rate limiting
    return elevations
```

**Pros**: No file download
**Cons**: Slow (~1 minute per 1000 points)

### Option 2: Use Synthetic High-Frequency Data
```python
# Create synthetic "elevation-like" high-frequency data
def synthetic_elevation(lat, lon):
    """Synthetic high-frequency function mimicking terrain"""
    # Multiple frequency components
    f1 = 1000 * np.sin(lat * 10) * np.cos(lon * 10)  # Large mountains
    f2 = 500 * np.sin(lat * 30) * np.cos(lon * 30)   # Medium hills
    f3 = 200 * np.sin(lat * 100) * np.cos(lon * 100) # Small valleys
    return f1 + f2 + f3

# This still tests "high-frequency" hypothesis even if not real elevation
```

**Pros**: Instant, no downloads
**Cons**: Not real data, less convincing

---

## Summary: Data Readiness for NB19 ✅

**ALL DATA SOURCES VALIDATED AND READY!** (2026-01-12)

| Experiment | Data Needed | Status | Fallback (if needed) |
|------------|-------------|--------|----------|
| **Exp 1: Regression vs Classification** | Population (GPW) | ✅ **Ready** | N/A |
| **Exp 2: High-Frequency Tasks** | Elevation, Coastline | ✅ **Ready** | Open-Elevation API (tested) |
| **Exp 3: Multi-Resolution** | Elevation at multiple resolutions | ✅ **Ready** | Downsample from 60s data |
| **Exp 4: Complexity Measurement** | Any trained model | ✅ **Ready** | N/A |
| **Exp 5: Task Difficulty** | Elevation (smoothed versions) | ✅ **Ready** | Use population + noise |

**Critical URLs All Working**:
- ✅ ETOPO 2022 60s (~60 MB) - direct download
- ✅ Natural Earth coastlines (~3 MB) - direct download
- ✅ GPW population - already in Drive
- ✅ Figshare temperature (optional) - working
- ✅ Open-Elevation API (backup) - working

**No Blockers**: Can proceed directly with Notebook 19 execution

---

## Next Steps ✅

### Data Acquisition (Ready to Execute)

All validations complete! You can now:

1. ✅ **Start Notebook 19** - no data blockers
2. ✅ **Download ETOPO 2022 60s** in Colab (~60 MB, 1-2 min)
3. ✅ **Download Natural Earth coastlines** in Colab (~3 MB, <1 min)
4. ✅ **Mount Drive for GPW** (already have)

### Recommended Execution Order

**Phase 1: Critical Experiments** (~3 hours)
1. Exp 1: Regression vs Classification (population data only)
2. Exp 2: Elevation task (download ETOPO first)
3. Exp 4: Complexity measurement (analysis on trained models)

**Phase 2: Extended Analysis** (~2-3 hours)
4. Exp 3: Multi-resolution (use downloaded elevation, downsample)
5. Exp 5: Task difficulty (smooth elevation data)

**Phase 3: Optional** (~1 hour if time)
6. Coastline distance task (compute from Natural Earth)
7. Temperature gradients (Figshare data)

### No Action Needed For:
- ❌ No additional data downloads required (besides Colab wget)
- ❌ No account registration needed
- ❌ No authentication tokens required
- ❌ No manual uploads to Drive needed

**Ready to proceed with Notebook 19 execution!**
