# Notebook 20: Data Sources for Regional Analysis

**Created**: 2026-01-12
**Purpose**: Document data acquisition for regional/continental analysis experiments

---

## Overview

NB20 requires multi-resolution, multi-region data to test if learned activations show advantages at local/regional scales (vs global scale tested in NB19).

**Key Requirements**:
1. Multiple spatial resolutions (30km → 2km → 1km)
2. Continental/regional subsets (not just global)
3. High-frequency features (coastlines, land cover boundaries)
4. Urban vs rural distinction

---

## Data Sources by Experiment

### Experiment 1: Continental Comparisons

**Data**: ETOPO 2022 Elevation (Already Available ✅)

**Source**: NOAA NCEI
- **URL**: https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc
- **Resolution**: 60 arc-seconds (~2 km at equator)
- **Coverage**: Global
- **Format**: NetCDF4
- **Size**: 478 MB
- **Status**: ✅ Already downloaded in NB19

**Regional Extraction**:
```python
import xarray as xr

# Load global data
ds = xr.open_dataset('etopo_60s.nc')

# Define regions
REGIONS = {
    'north_america': {'lat': slice(50, 25), 'lon': slice(-125, -65)},
    'europe': {'lat': slice(70, 35), 'lon': slice(-10, 40)},
    'asia_himalayas': {'lat': slice(40, 25), 'lon': slice(70, 100)},
    'africa_sahara': {'lat': slice(30, 15), 'lon': slice(-10, 30)},
    'south_america_andes': {'lat': slice(10, -40), 'lon': slice(-80, -60)},
}

# Extract regional subset
region_ds = ds.sel(lat=REGIONS['north_america']['lat'],
                   lon=REGIONS['north_america']['lon'])
```

**Terrain Classification**:
- **Mountainous**: Himalayas (Asia), Andes (S. America), Rockies (N. America)
- **Flat**: Sahara (Africa), Great Plains (N. America), European Plain
- **Mixed**: Europe (Alps + plains)

---

### Experiment 2: SH Encoding Levels (L=10/20/40)

**Data**: Same ETOPO elevation + GPW population

**No additional data needed** - testing different encoding parameters on existing data.

**SH Dimensionality**:
- **L=10**: 121 dimensions (current baseline)
- **L=20**: 441 dimensions (medium)
- **L=40**: 1681 dimensions (high)

**Computational Considerations**:
- L=40 with 3×256 network = ~5M parameters (manageable)
- May need to reduce hidden_dim to 128 for L=40
- Training time: L=40 likely 2-3× slower than L=10

---

### Experiment 3: Spatial Resolution Within Regions

**Requires NEW high-resolution data**

#### Option A: SRTM 30m (Recommended)

**Source**: NASA SRTM (Shuttle Radar Topography Mission)
- **URL**: https://earthexplorer.usgs.gov/ (requires free account)
- **Alternative**: OpenTopography (https://opentopography.org/)
- **Resolution**: 1 arc-second (~30 meters)
- **Coverage**: 60°N to 56°S (most land areas)
- **Format**: GeoTIFF
- **Size per tile**: 1° × 1° = ~25 MB compressed

**Regional Downloads** (Priority tiles):
1. **North America (Rockies)**:
   - Tiles: N40W106 to N39W105 (Colorado Rockies)
   - Area: ~100km × 100km
   - Reason: High relief, clear peaks/valleys

2. **Asia (Himalayas)**:
   - Tiles: N28E086 to N27E087 (Everest region)
   - Area: ~100km × 100km
   - Reason: Highest relief on Earth

3. **Europe (Alps)**:
   - Tiles: N46E007 to N45E008 (Mont Blanc)
   - Area: ~100km × 100km
   - Reason: Moderate relief, diverse terrain

**Estimated Download**: 3 regions × 4 tiles = 12 tiles × 25 MB = ~300 MB

**Python Access** (OpenTopography API):
```python
import rasterio
from rasterio.merge import merge
import requests

def download_srtm_tile(lat, lon, output_dir):
    """Download SRTM 1-arc-second tile"""
    # OpenTopography API endpoint
    url = f"https://portal.opentopography.org/API/globaldem"
    params = {
        'demtype': 'SRTMGL1',  # 1 arc-second
        'south': lat,
        'north': lat + 1,
        'west': lon,
        'east': lon + 1,
        'outputFormat': 'GTiff',
    }
    # Requires API key (free registration)
    response = requests.get(url, params=params)
    # ... download and save
```

#### Option B: ASTER GDEM (Alternative)

**Source**: NASA/METI ASTER
- **URL**: https://search.earthdata.nasa.gov/
- **Resolution**: 1 arc-second (~30 meters)
- **Coverage**: 83°N to 83°S (more than SRTM)
- **Format**: GeoTIFF
- **Size**: Similar to SRTM

**Pros**: Better coverage (high latitudes)
**Cons**: More noise than SRTM, requires EarthData login

#### Multi-Resolution Strategy

**For each region, create 3 resolutions**:

1. **Coarse (30 km)**: Downsample ETOPO or SRTM by 1000×
   - ~1000 samples per region

2. **Medium (2 km)**: Use ETOPO 60s directly
   - ~2500 samples per 100×100 km region

3. **Fine (1 km)**: Downsample SRTM 30m by 30×
   - ~10,000 samples per 100×100 km region

---

### Experiment 4: Urban vs Rural Patterns

**Data**: GPW Population Density (Already Available ✅) + OpenStreetMap (Optional)

#### Primary: GPW Population Density
- **Status**: ✅ Already loaded in 19b
- **Resolution**: 15 arc-minutes (~30 km)
- **Coverage**: Global
- **Use**: Identify urban vs rural based on density thresholds

**Urban Classification**:
```python
# Classify regions by population density
urban = pop_data > 1000  # > 1000 people/km²
suburban = (pop_data > 100) & (pop_data <= 1000)
rural = pop_data <= 100
```

**Target Regions**:
- **Dense Urban**: Tokyo (35.6°N, 139.7°E), NYC (40.7°N, -74.0°W), Mumbai (19.1°N, 72.9°E)
- **Suburban**: US suburbs, European towns
- **Rural**: Agricultural areas, wilderness

#### Optional: OpenStreetMap Building Footprints

**Source**: OpenStreetMap via Overpass API
- **URL**: https://overpass-api.de/
- **Resolution**: Vector data (exact building boundaries)
- **Coverage**: Global (quality varies)
- **Format**: GeoJSON, XML
- **Use**: Sharp urban boundaries (building presence/absence)

**Query Example**:
```python
import requests

def get_buildings(bbox):
    """Get building footprints from OSM
    bbox: (south, west, north, east)
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out geom;
    """
    response = requests.get(overpass_url, params={'data': query})
    return response.json()
```

**Decision**: START with GPW only, add OSM if time permits

---

### Experiment 5: Boundary-Rich Tasks

#### 5a. Coastline Distance (Already Have Data ✅)

**Source**: Natural Earth
- **Status**: ✅ Downloaded in NB19
- **URL**: https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip
- **Resolution**: 10m (1:10 million scale)
- **Coverage**: Global
- **Format**: Shapefile
- **Size**: 3 MB
- **Features**: 4,133 coastline segments

**Usage**:
```python
import geopandas as gpd
from scipy.spatial import cKDTree

# Load coastlines
coastlines = gpd.read_file('ne_10m_coastline.shp')

# Compute distance from any point to nearest coastline
def distance_to_coast(lat, lon):
    # Extract coastline coordinates
    coast_coords = np.vstack([geom.coords for geom in coastlines.geometry])

    # Build KD-tree
    tree = cKDTree(coast_coords)

    # Query distance
    dist, _ = tree.query([lat, lon])
    return dist
```

**Task**: Predict distance to coast (step function at coastline)

#### 5b. Land Cover Transitions (NEW DATA NEEDED)

**Option A: ESA CCI Land Cover (Recommended)**

**Source**: ESA Climate Change Initiative
- **URL**: https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover
- **Resolution**: 300 meters
- **Coverage**: Global
- **Format**: NetCDF or GeoTIFF
- **Size**: ~2 GB for global annual map
- **Classes**: 22 land cover types (forest, cropland, urban, water, etc.)

**Access** (requires Copernicus account):
```python
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'satellite-land-cover',
    {
        'variable': 'all',
        'format': 'zip',
        'year': '2020',
        'version': 'v2.1.1',
    },
    'esacci-lc-2020.zip'
)
```

**Alternative: Direct Download**: https://maps.elie.ucl.ac.be/CCI/viewer/download.php

**Use Case**: Identify boundaries between:
- Forest ↔ Cropland
- Urban ↔ Rural
- Desert ↔ Vegetation
- Land ↔ Water

**Processing**:
```python
import rasterio

# Load land cover
lc = rasterio.open('ESACCI-LC-2020.tif')

# Detect edges (boundaries)
from scipy.ndimage import sobel
edges = sobel(lc.read(1))

# Sample near high-edge regions (boundaries)
boundary_mask = edges > threshold
```

**Option B: MODIS Land Cover (Alternative)**

**Source**: NASA MODIS MCD12Q1
- **URL**: https://lpdaac.usgs.gov/products/mcd12q1v061/
- **Resolution**: 500 meters
- **Coverage**: Global
- **Format**: HDF or GeoTIFF
- **Size**: ~1 GB per year
- **Classes**: 17 IGBP classes

**Pros**: Well-validated, widely used
**Cons**: Coarser than ESA CCI (500m vs 300m)

#### 5c. Elevation Gradients (Use SRTM from Exp 3)

**Data**: SRTM 30m (from Exp 3)

**Task**: Predict elevation gradient magnitude (peaks/valleys)

**Computation**:
```python
from scipy.ndimage import sobel

# Compute gradients
grad_x = sobel(elevation, axis=1)
grad_y = sobel(elevation, axis=0)

# Gradient magnitude
gradient = np.sqrt(grad_x**2 + grad_y**2)

# Task: Predict gradient from coordinates
# High gradients = sharp features (peaks, valleys, cliffs)
```

---

## Data Priority & Timeline

### Phase 1: Immediate (Use Existing Data)
**Experiments 1 & 2**: ✅ No new downloads needed
- Exp 1 (Continental): Use ETOPO elevation
- Exp 2 (SH levels): Use ETOPO + GPW
- **Timeline**: Can start immediately

### Phase 2: High Priority (Essential for Exp 3)
**SRTM 30m Elevation**: Download 3 regional tiles
- **Regions**: Rockies, Himalayas, Alps
- **Size**: ~300 MB
- **Source**: OpenTopography (free, no account needed for small areas)
- **Timeline**: ~30 min download

### Phase 3: Medium Priority (Exp 4 & 5)
**ESA CCI Land Cover**: For boundary tasks
- **Size**: ~2 GB global or ~200 MB regional
- **Source**: Copernicus (requires free registration)
- **Timeline**: ~1 hour download + registration

### Phase 4: Optional (If Time Permits)
**OpenStreetMap Buildings**: For urban boundaries
- **Access**: API queries (no bulk download)
- **Timeline**: Queries on-demand during experiment

---

## Recommended Download Order

1. **Start with existing data** (Exp 1 & 2)
   - Test continental differences with ETOPO
   - Compare SH encoding levels
   - Establish baseline before adding complexity

2. **Download SRTM tiles** (for Exp 3)
   - 3 regions × 4 tiles = 12 downloads
   - Can run in parallel with Exp 1/2

3. **ESA Land Cover** (for Exp 5b)
   - Download while running earlier experiments
   - Can skip if SRTM results are definitive

4. **OSM Buildings** (optional)
   - Only if urban/rural analysis shows promise
   - Query-based, so no upfront download needed

---

## Storage Requirements

**Total Estimated Storage**:
- ETOPO elevation: 478 MB (already have)
- GPW population: ~50 MB (already have)
- SRTM 30m tiles: ~300 MB (new)
- ESA CCI land cover: ~2 GB global or ~200 MB regional (new)
- **Total new data**: ~500 MB to 2.3 GB

**Recommendation**: Download regional subsets where possible to minimize storage

---

## Data Access Tools

### Python Libraries
```python
# Geospatial I/O
import rasterio           # Read/write raster data (GeoTIFF, etc.)
import xarray as xr       # NetCDF, multi-dimensional arrays
import geopandas as gpd   # Vector data (shapefiles)

# Downloading
import requests           # HTTP downloads
import cdsapi             # Copernicus Climate Data Store

# Processing
import numpy as np
from scipy.ndimage import zoom  # Resampling
from scipy.spatial import cKDTree  # Distance calculations
```

### Web Services
- **OpenTopography**: SRTM downloads, no account needed for small areas
- **Copernicus Climate Data Store**: ESA CCI land cover, requires free account
- **EarthExplorer (USGS)**: Alternative for SRTM/ASTER, requires free account
- **Overpass API**: OpenStreetMap queries, no account needed

---

## Data Validation Checklist

Before running experiments, verify:

✅ **Coordinate Systems**:
- All data in same CRS (WGS84 lat/lon preferred)
- Check with `rasterio.crs` or `geopandas.crs`

✅ **Resolution Consistency**:
- Verify arc-seconds vs meters
- Check actual grid spacing (not just metadata)

✅ **NoData Handling**:
- Identify nodata values (-9999, -3.4e38, etc.)
- Mask before sampling

✅ **Regional Bounds**:
- Confirm lat/lon ranges match intended regions
- Check for off-by-one errors in slicing

✅ **Sample Size**:
- Enough valid pixels per region (target: 5000+)
- Balance train/test with spatial blocking

---

## Next Steps

1. ✅ Create this documentation
2. ⏳ Download SRTM tiles for 3 priority regions (Exp 3)
3. ⏳ Register for Copernicus if doing land cover (Exp 5b)
4. ⏳ Create data loading utilities in NB20
5. ⏳ Validate all data sources before experiments

---

## Notes

- **Focus on efficiency**: Start with existing data, add high-res only if needed
- **Regional first**: Continental comparisons may be sufficient before going to fine resolution
- **Iterative approach**: If Exp 1 shows no advantage, may not need Exp 3-5
- **Publication quality**: Document all data sources, versions, access dates

**Ready to implement once user confirms priority order.**
