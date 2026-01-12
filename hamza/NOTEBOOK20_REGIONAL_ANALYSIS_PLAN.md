# Notebook 20: Regional Analysis & SH Encoding Study

**Created**: 2026-01-12
**Status**: Planning
**Motivation**: Global experiments (NB19) showed minimal spline advantage. Test if regional/local tasks show different patterns.

---

## Hypothesis

**Global tasks may obscure local patterns where learned activations excel.**

Rationale:
1. All NB19 tests were global-scale (entire planet or large samples)
2. SH(L=10) = 121 dimensions may be over-smoothing regional details
3. High-frequency geographic features are more prominent at regional scales
4. Paper (Teney et al.) tested on image patches, not global data

---

## Proposed Experiments

### Experiment 1: Continental Comparisons

**Test elevation prediction across different continents**

Regions to test:
- **North America** (mountainous west, flat midwest)
- **Europe** (Alps, Scandinavia, plains)
- **Africa** (diverse: deserts, mountains, rift valleys)
- **Asia** (Himalayas - highest frequency)
- **South America** (Andes)

**Hypothesis**:
- Mountainous regions (Asia/Himalayas, Americas/Andes) favor splines
- Flat regions (plains, deserts) favor ReLU (simplicity bias sufficient)

**Metrics**: R² per region, compare Spline vs ReLU advantage

---

### Experiment 2: SH Encoding Dimensionality

**Compare L=10, L=20, L=40 spherical harmonics**

Setup:
- Same regional tasks as Exp 1
- Test each region with 3 SH levels:
  - **L=10** (121 dims) - current baseline
  - **L=20** (441 dims) - medium resolution
  - **L=40** (1681 dims) - high resolution

**Hypothesis**:
- Lower L (more smoothing) → smaller spline advantage
- Higher L (less smoothing) → larger spline advantage
- There's an optimal L where splines add value without overfitting

**Interaction hypothesis**:
- Flat regions: L doesn't matter (already smooth)
- Mountainous regions: Higher L + Spline shows advantage

---

### Experiment 3: Spatial Resolution Scaling

**Test at multiple data resolutions within regions**

Resolutions:
- **Coarse**: 15 arc-min (~30 km) - GPW population resolution
- **Medium**: 1 arc-min (~2 km) - ETOPO standard resolution
- **Fine**: 30 arc-sec (~1 km) - High-res SRTM data

Regions: North America (varied terrain) and Asia (Himalayas)

**Hypothesis**:
- Coarse resolution: ReLU sufficient (smooth signals)
- Fine resolution: Spline advantage emerges (captures local variation)
- Effect stronger in mountainous regions

---

### Experiment 4: Urban vs Rural Patterns

**Test population density in different settlement types**

Regions:
- **Dense urban**: Tokyo, New York, Mumbai (sharp boundaries)
- **Suburban**: Sprawling cities (gradual transitions)
- **Rural**: Agricultural/wilderness (smooth gradients)

**Hypothesis**:
- Urban (step functions) → strong spline advantage
- Rural (smooth gradients) → ReLU sufficient
- This directly tests "high-frequency" assumption with human-made patterns

---

### Experiment 5: Boundary-Rich Tasks

**Focus on geographic features with sharp transitions**

Tasks:
1. **Coastline distance** (step function at land-water boundary)
2. **Elevation gradient magnitude** (peaks/valleys)
3. **Land cover transitions** (forest-desert, urban-rural boundaries)

Regions: Coastal areas, mountain ranges, ecotones

**Hypothesis**:
- Sharp boundaries are the "alpha" we've been looking for
- These should show largest spline advantage
- If splines don't win here, they won't win anywhere

---

## Data Requirements

### Already Available
- ✅ ETOPO elevation (global, 60s resolution)
- ✅ GPW population (global, 15 arc-min)
- ✅ Natural Earth coastlines

### New Data Needed
- **SRTM 30m**: High-res elevation (regional downloads)
- **Land cover**: ESA CCI or MODIS (for boundary tasks)
- **Urban boundaries**: OpenStreetMap building footprints

---

## Implementation Plan

### Architecture Setup

```python
def create_regional_encoder(region_name, sh_level=10, activation='spline'):
    """
    region_name: 'north_america', 'asia', 'europe', etc.
    sh_level: 10, 20, or 40
    activation: 'relu', 'spline', 'siren'
    """
    return UniversalEncoder(
        input_type='sh',
        sh_legendre_polys=sh_level,
        activation_type=activation,
        activation_kwargs={'n_knots': 15, 'init': 'relu'}
    )
```

### Regional Sampling Strategy

```python
def sample_region(data, lons, lats, region_bounds, n_samples=5000):
    """
    region_bounds: {'lat_min': X, 'lat_max': Y, 'lon_min': A, 'lon_max': B}
    """
    lat_mask = (lats >= region_bounds['lat_min']) & (lats <= region_bounds['lat_max'])
    lon_mask = (lons >= region_bounds['lon_min']) & (lons <= region_bounds['lon_max'])
    # ... spatial blocking within region
```

### Region Definitions

```python
REGIONS = {
    'north_america': {'lat_min': 25, 'lat_max': 50, 'lon_min': -125, 'lon_max': -65},
    'europe': {'lat_min': 35, 'lat_max': 70, 'lon_min': -10, 'lon_max': 40},
    'asia_himalayas': {'lat_min': 25, 'lat_max': 40, 'lon_min': 70, 'lon_max': 100},
    'africa_sahara': {'lat_min': 15, 'lat_max': 30, 'lon_min': -10, 'lon_max': 30},
    'south_america_andes': {'lat_min': -40, 'lat_max': 10, 'lon_min': -80, 'lon_max': -60},
}
```

---

## Expected Outcomes

### Scenario A: Regional Advantage Found
- Splines excel in mountainous/urban/boundary-rich regions
- SH level matters: Higher L with splines shows clear benefit
- **Alpha found**: Use splines for high-frequency regional tasks

### Scenario B: No Regional Advantage
- ReLU wins or ties across all regions/resolutions
- SH encoding already captures relevant patterns
- **Conclusion**: SH + ReLU is the right baseline for geographic data

### Scenario C: Task-Specific Patterns
- Urban/coastal tasks show spline advantage
- Smooth tasks (plains, rural) show ReLU advantage
- **Guidance**: Task-dependent architecture selection

---

## Integration with Existing Roadmap

**Original Plan**:
- NB19: Simplicity bias tests (global) ✅
- NB20: Visualization
- NB21: Robustness
- NB22: Architecture variations
- NB23: Raw + Learned encoding

**Revised Plan**:
- NB19: Simplicity bias tests (global) ✅
- **NB20: Regional Analysis & SH Encoding** ← NEW (this notebook)
- NB21: Visualization (can now visualize regional patterns)
- NB22: Robustness (test regional findings across seeds)
- NB23: Architecture variations (informed by regional results)
- NB24: Raw + Learned encoding

**Why this order makes sense**:
1. NB20 tests a key hypothesis (scale matters) before moving to secondary analyses
2. Visualization (NB21) benefits from regional findings to show
3. Robustness (NB22) can validate regional discoveries
4. Architecture (NB23) can be targeted to promising regions/tasks

---

## Success Metrics

**Minimum viable result**:
- Complete Exp 1 (continental comparison) and Exp 2 (SH encoding levels)
- Determine if region/encoding matters for spline advantage

**Full success**:
- All 5 experiments completed
- Clear guidance on when to use splines vs ReLU
- Either find "alpha" or conclusively show SH+ReLU is sufficient

**Publication threshold**:
- Find >5% spline advantage in specific region/task/encoding combination, OR
- Comprehensive negative result with clear mechanistic explanation

---

## Timeline Estimate

- **Data acquisition**: 2-3 hours (SRTM downloads, land cover)
- **Exp 1 (Continental)**: 3 hours (5 regions × 3 activations × 100 epochs)
- **Exp 2 (SH levels)**: 6 hours (5 regions × 3 SH levels × 3 activations)
- **Exp 3 (Resolution)**: 4 hours (2 regions × 3 resolutions × 3 activations)
- **Exp 4 (Urban/Rural)**: 3 hours (3 region types × 3 activations)
- **Exp 5 (Boundaries)**: 3 hours (3 tasks × 3 activations)

**Total runtime**: ~20 hours on Colab T4 GPU (can split into multiple sessions)

---

## Open Questions

1. **SH encoding on regional data**: Do we center SH on region centroid, or keep global coordinates?
   - Recommendation: Keep global coords for consistency, but test both

2. **Sample size per region**: 5,000 samples sufficient? Or need more for small regions?
   - Recommendation: Start with 5,000, increase if variance is high

3. **Spatial blocking within regions**: 5° grid too coarse for small regions?
   - Recommendation: Use adaptive grid (1° for small regions, 5° for large)

4. **Computational cost of L=40**: 1681 dimensions feasible with 3-layer network?
   - Recommendation: May need to reduce hidden_dim (256→128) for L=40

---

## Next Steps

1. ✅ Create this planning document
2. ⏳ Get user confirmation on priorities (all experiments or subset?)
3. ⏳ Decide on data downloads (SRTM, land cover, or skip Exp 3/5?)
4. ⏳ Create NB20 Jupyter notebook with selected experiments
5. ⏳ Run and analyze results
6. ⏳ Update ANALYSIS_NOTEBOOK19.md with regional findings

---

## Notes

- This addresses the key limitation of NB19: all tests were global-scale
- Regional analysis is natural next step after finding minimal global advantage
- SH encoding comparison (L=10 vs L=40) directly tests encoding hypothesis
- If this also shows no advantage, we have very strong evidence that SH+ReLU is optimal for geographic data
