# Notebook 19: Data Reference

**Quick reference for data used in NB19 experiments.**

---

## Data Summary

| Dataset | Source | Size | Used In | Status |
|---------|--------|------|---------|--------|
| **ETOPO 2022 Elevation** | NOAA | 478 MB | Exp 2, 3, 5 | ✅ Downloaded |
| **Natural Earth Coastlines** | NaturalEarth | 3 MB | Future (coastline distance) | ✅ Downloaded |
| **GPW Population** | CIESIN/Drive | - | Exp 1 (failed) | ❌ Path issue |
| **Temperature** (optional) | Figshare | 375 KB | Future | ✅ Downloaded |

---

## ETOPO 2022 Elevation

**URL**: https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc

**Details**:
- Resolution: 60 arc-seconds (~2 km at equator)
- Coverage: Global (Northern Hemisphere file used)
- Format: NetCDF4
- Size: 478.3 MB downloaded

**Loading**:
```python
import xarray as xr
ds = xr.open_dataset('etopo_60s.nc')
elevation = ds['z'].values  # Shape: (10800, 21600)
lats = ds['lat'].values
lons = ds['lon'].values
```

**Used In**:
- Exp 2: High-frequency task (elevation prediction)
- Exp 3: Multi-resolution (downsampled to 3 resolutions)
- Exp 5: Task difficulty (smoothed versions)

---

## Natural Earth Coastlines

**URL**: https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip

**Details**:
- Resolution: 10m (detailed)
- Coverage: Global
- Format: Shapefile
- Size: 3 MB

**Loading**:
```python
import geopandas as gpd
coastlines = gpd.read_file('coastline_data/ne_10m_coastline.shp')
# 4133 features loaded
```

**Intended Use**: Coastline distance task (step functions)
**Status**: Downloaded but not used yet in experiments

---

## GPW Population (Issue)

**Expected Path**: `/content/drive/MyDrive/grad/learned_activations/dataverse_files.zip`

**Problem**: File structure mismatch
- Script looks for: `gpw_v4_population_density_rev11_2020_15_min.tif`
- Zip may have different structure or nested folders

**Solution**: See supplementary notebook for corrected extraction

**Details**:
- Resolution: 15 arc-minutes (~30 km)
- Coverage: Global
- Format: GeoTIFF
- Source: CIESIN GPW v4

---

## Temperature Data (Optional)

**URL**: https://springernature.figshare.com/ndownloader/files/12609182

**Details**:
- Observations: 3,076 locations
- Format: CSV
- Size: 375 KB

**Loading**:
```python
import pandas as pd
temp_df = pd.read_csv('temperature.csv')
# Contains: latitude, longitude, temperature columns
```

**Intended Use**: Medium-frequency task (temperature gradients)
**Status**: Downloaded, not yet used

---

## Archived Documentation

More detailed documentation moved to `Archive/NB19/`:
- `NOTEBOOK19_DATA_SOURCES.md` - Comprehensive data guide
- `NOTEBOOK19_DATA_QUICKSTART.py` - Acquisition script
- `NOTEBOOK19_READY.md` - Launch checklist

These files are preserved for reference but not needed for day-to-day work.

---

**See Also**:
- [NB19_DATA_VALIDATION_SUMMARY.md](NB19_DATA_VALIDATION_SUMMARY.md) - Detailed validation results
- [ANALYSIS_NOTEBOOK19.md](ANALYSIS_NOTEBOOK19.md) - Experimental results
