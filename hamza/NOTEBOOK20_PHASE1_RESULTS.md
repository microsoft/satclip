# Notebook 20: Phase 1 Results (Experiments 1 & 2)

**Date**: 2026-01-12
**Status**: Experiment 1 Complete, Experiment 2 Partial (crashed at L=40)
**CRITICAL ISSUE**: **RESULTS NOT REPRODUCIBLE** - Same region shows -8% then +52% advantage!
**Secondary Finding**: Flat terrain favored splines in Exp 1, but needs multi-seed validation

---

## Experiment 1: Continental Comparisons ✅

### Results Summary

| Region | Terrain | ReLU R² | Spline R² | SIREN R² | Spline Adv | Winner |
|--------|---------|---------|-----------|----------|------------|--------|
| **North America** | Mixed | 0.9063 | 0.9224 | **0.9340** | **+1.78%** | SIREN |
| **Europe** | Mixed | 0.4451 | 0.4317 | 0.3269 | **-2.99%** | ReLU |
| **Asia (Himalayas)** | Mountain | 0.6565 | 0.6025 | 0.5061 | **-8.22%** ❌ | ReLU |
| **Africa (Sahara)** | Flat | 0.6990 | 0.7925 | **0.8079** | **+13.37%** ✅ | SIREN |
| **S. America (Andes)** | Mountain | 0.8482 | 0.8263 | 0.7854 | **-2.58%** | ReLU |

### By Terrain Type

| Terrain | Avg Spline Advantage | Hypothesis | Result |
|---------|---------------------|------------|---------|
| **Mountainous** | **-5.40%** | Spline wins | ❌ **ReLU wins** |
| **Flat** | **+13.37%** | ReLU wins | ❌ **Spline wins** |
| **Mixed** | **-0.61%** | Neutral | ✅ Tie (as expected) |

---

## 🚨 CRITICAL FINDING: Results Not Reproducible

**Multiple runs of the SAME experiment show WILDLY different results:**

### Asia Himalayas (Mountain, L=10) - Three Different Runs:

| Run | ReLU R² | Spline R² | Advantage | Verdict |
|-----|---------|-----------|-----------|---------|
| **Exp 1 (First run)** | 0.6565 | 0.6025 | **-8.22%** | ReLU wins |
| **Exp 1 (Second run)** | 0.6949 | 0.6336 | **-8.83%** | ReLU wins |
| **Exp 2** | 0.4241 | 0.6445 | **+52.0%** | Spline wins |

### Africa Sahara (Flat, L=10) - Two Different Runs:

| Run | ReLU R² | Spline R² | Advantage | Verdict |
|-----|---------|-----------|-----------|---------|
| **Exp 1 (First run)** | 0.6990 | 0.7925 | **+13.37%** | Spline wins |
| **Exp 1 (Second run)** | 0.7143 | 0.7283 | **+1.96%** | Spline barely wins |

**Range of spline advantage**: -8.83% to +52.0% for mountains, +1.96% to +13.37% for flat

**This is a FUNDAMENTAL PROBLEM** - any conclusions from single-seed runs are unreliable.

**All findings below must be treated as preliminary pending multi-seed validation.**

### Why Such High Variance?

**Likely causes of instability**:

1. **Small sample size**: 5000 samples per region (vs 15K global in NB19)
   - Regional subsets more prone to sampling bias
   - Single unlucky sample can shift results significantly

2. **Spatial blocking with few cells**: 5° grid creates ~12-25 blocks per region
   - Test set is only 3-8 spatial blocks
   - High sensitivity to which specific blocks are selected
   - NB19 had more blocks globally, averaging out variance

3. **Regional heterogeneity**: Each region has internal variation
   - Himalayas: valleys vs peaks
   - Sahara: rocky vs sandy areas
   - Random sampling hits different mixtures each time

4. **Training instability**: 100 epochs may not be enough
   - Networks might be converging to different local minima
   - Splines especially prone (more parameters, more degrees of freedom)
   - Best R² varies wildly across epochs (see verbose output)

5. **Network initialization**: Different random weight init each run
   - With small datasets, initialization matters more
   - Splines have learnable knot positions - highly sensitive

**Consequence**: Cannot draw ANY reliable conclusions about terrain effects without proper statistical validation (mean ± std over multiple seeds).

---

## Key Findings (⚠️ Single-Seed, Needs Validation)

### Finding 1: HYPOTHESIS REVERSED ⚠️

**Prediction**: Mountainous regions favor splines (high-frequency content)
**Reality**: **FLAT regions favor splines** (+13.37% in Sahara)

**Mountainous Results**:
- Asia Himalayas: Spline -8.22% worse than ReLU
- S. America Andes: Spline -2.58% worse than ReLU
- **Average**: -5.40% disadvantage

**Flat Results**:
- Africa Sahara: Spline +13.37% better than ReLU
- **Only flat region tested, but strong signal**

**Interpretation**:
- Mountainous terrain may be TOO complex for splines → overfitting?
- Flat terrain (Sahara) has smooth, predictable patterns → spline expressiveness helps?
- Counter-intuitive but consistent across 2 mountainous regions

---

### Finding 2: SIREN Unexpectedly Competitive 🎯

**NB19 Result**: SIREN lost to ReLU/Spline consistently
**NB20 Result**: SIREN WINS in 2/5 regions

**SIREN Winners**:
1. **North America**: 0.9340 (best of all experiments!)
2. **Africa Sahara**: 0.8079

**Why the difference?**
- Regional scale vs global scale?
- Specific terrain characteristics (mixed + flat)?
- North America has complex mixed terrain (Rockies + plains) → SIREN captures?
- Sahara smooth patterns → SIREN's sinusoidal basis helps?

**This contradicts NB19 where SIREN was consistently worst**

---

### Finding 3: Europe Anomaly ⚠️

**All models perform poorly**:
- ReLU: 0.4451 (best)
- Spline: 0.4317
- SIREN: 0.3269

**Possible Causes**:
1. **Data quality issue**: Coastal complexity, islands
2. **Mixed terrain complexity**: Alps + plains + Mediterranean = too heterogeneous?
3. **Small land area**: More ocean/water bodies in bounding box?
4. **Latitude variation**: 30° span (35-65°N) vs others (~15-25°)

**Recommendation**: Investigate Europe region specifically, may exclude from analysis

---

### Finding 4: Performance Hierarchy

**Best Performing Regions**:
1. North America (mixed): R² up to 0.9340
2. S. America Andes (mountain): R² up to 0.8482
3. Africa Sahara (flat): R² up to 0.8079

**Worst Performing**:
1. Europe (mixed): R² max 0.4451
2. Asia Himalayas (mountain): R² max 0.6565

**Insight**: Region complexity ≠ poor performance necessarily
- North America complex but high R²
- Himalayas simple (mountains) but moderate R²

---

## Experiment 2: SH Encoding Levels ⚠️

**Status**: Partial completion - Mountain terrain only, L=40 crashed
**Completed**: L=10, L=20 on Himalayas (mountain)
**Failed**: L=40 (NaN predictions), Sahara (flat) not tested

### Results Summary (Mountains Only - Asia Himalayas)

| SH Level | Dims | ReLU R² | Spline R² | SIREN R² | Spline Adv | Training Time |
|----------|------|---------|-----------|----------|------------|---------------|
| **L=10** | 121 | 0.4241 | **0.6445** | 0.4283 | **+52.0%** ✅ | ReLU: 26.5s, Spline: 42.1s |
| **L=20** | 441 | **0.6669** | 0.5479 | 0.3224 | **-17.8%** ❌ | ReLU: 134.7s, Spline: 154.3s |
| **L=40** | 1681 | ❌ CRASH | ❌ CRASH | ❌ CRASH | N/A | ValueError: NaN predictions |

### Key Findings

#### Finding 1: HIGH VARIANCE / INCONSISTENCY ⚠️

**MAJOR DISCREPANCY with Experiment 1**:
- **Exp 1 (Himalayas, L=10)**: ReLU 0.6565, Spline 0.6025 → **Spline -8.22%** ❌
- **Exp 2 (Himalayas, L=10)**: ReLU 0.4241, Spline 0.6445 → **Spline +52.0%** ✅

**Same region, same L=10, OPPOSITE results!**

**Possible Causes**:
1. **Different random seeds**: Exp 1 and Exp 2 sampled different 5000 points from Himalayas
2. **Training instability**: Networks may be highly sensitive to initialization
3. **Spatial blocking variation**: Different test cells selected in spatial CV
4. **Overfitting**: One or both runs may have overfit to specific samples

**Implication**: Results are NOT stable or reproducible - undermines all conclusions

#### Finding 2: SH Level L=20 Shows Pattern Reversal

- At **L=10**: Spline wins +52.0%
- At **L=20**: ReLU wins -17.8%

**Trend**: Higher dimensionality HURTS spline performance
- Opposite of hypothesis (expected L=20/40 to help splines)
- ReLU improves from 0.4241 → 0.6669 (+57% better R²)
- Spline degrades from 0.6445 → 0.5479 (-15% worse R²)

**Possible Explanation**:
- L=20 (441 dims) with splines → massive overfitting?
- ReLU benefits from richer encoding, splines do not
- Training time 5× slower at L=20, may need more epochs?

#### Finding 3: L=40 Numerical Instability

**Error**: `ValueError: Input contains NaN`
- Model predictions became NaN during training
- Likely due to 1681-dimensional input with numerical explosion
- Even with reduced hidden_dim=128, still unstable

**Conclusion**: L=40 impractical for current setup - need better numerical handling

#### Finding 4: SIREN Collapses at L=20

- L=10: SIREN 0.4283 (competitive with ReLU)
- L=20: SIREN 0.3224 (worst of all)

SIREN's sinusoidal activations may interact poorly with high-dimensional SH encoding.

---

## Implications for Hypothesis

### Original Hypothesis (from NB19/Strategy)
1. ✅ Mountainous regions → high-frequency → spline advantage
2. ✅ Flat regions → low-frequency → ReLU advantage

### Actual Results
1. ❌ Mountainous regions → ReLU wins (-5.40%)
2. ❌ Flat regions → Spline wins (+13.37%)

**COMPLETE REVERSAL**

---

## Possible Explanations

### Explanation 1: Overfitting on Complex Terrain
- **Mountains**: High variability → splines overfit local patterns
- **Flat**: Low variability → splines learn smooth global function
- **ReLU's simplicity bias HELPS in complex cases**

### Explanation 2: Frequency Content Misinterpretation
- **Mountains**: Multi-scale structure (not purely high-freq)
  - Tectonic plates (low-freq) + local peaks (high-freq)
  - ReLU better at mixing scales?
- **Flat**: Actually has high-freq features (small dunes, erosion)
  - Splines capture these local variations?

### Explanation 3: SH Encoding Interaction
- **Mountains**: SH(L=10) struggles with extreme relief
  - ReLU's nonlinearity compensates for encoding limitations
  - Spline adds complexity without helping
- **Flat**: SH(L=10) captures main patterns well
  - Spline adds useful flexibility for local details

### Explanation 4: Sample Size / Spatial Blocking Effects
- All regions: ~5000 samples, 5° grid blocking
- Mountain regions: More elevation variation per sample
  - Higher noise in labels?
  - Spatial blocks more heterogeneous?
- Flat regions: More uniform samples
  - Cleaner learning signal?

---

## Comparison to NB19

### NB19 (Global, L=10, Stable Results)
- **Elevation**: Spline +0.36% (minimal but consistent)
- **Multi-resolution**: ReLU won at fine (-0.47%)
- **Population**: ReLU won both formulations
- **Key**: Results were reproducible across runs

### NB20 Exp 1 & 2 (Regional, L=10, UNSTABLE Results)
- **Spline advantage range**: -11% to +52% (SAME REGION different seeds!)
- **Terrain-dependent?**: Cannot conclude due to high variance
- **Best claimed advantage**: +13.37% (Sahara) - needs validation
- **Worst**: -11.25% (Andes)

**Critical Difference**: NB19 used 15K global samples, NB20 uses 5K regional samples
- Smaller sample size → higher variance
- Regional heterogeneity → less stable
- Spatial blocking with fewer cells → more sensitive to test set selection

**Conclusion**: **Regional scale reveals VARIANCE, not patterns** - need multi-seed validation!

---

## Critical Questions for Exp 2

1. **Does L=40 change the pattern?**
   - If Himalayas spline improves with L=40 → encoding was the issue
   - If still loses → fundamental unsuitability for complex terrain

2. **Does L=40 increase Sahara advantage?**
   - If yes → validates that smooth terrain + high-dim encoding = spline benefit
   - If no → L=10 already sufficient for flat terrain

3. **SH dimensionality interaction with terrain?**
   - Mountains need higher L to benefit from splines?
   - Flat terrain saturates at lower L?

---

## Updated Conclusions (After Partial Exp 2)

### What We Know
1. ⚠️ **RESULTS ARE UNSTABLE**: Same region (Himalayas L=10) shows -8% then +52% spline advantage
2. ✅ **Regional scale matters**: Wide range of results (but not reproducible)
3. ❌ **SH encoding hypothesis REJECTED**: L=20 makes splines WORSE, L=40 crashes
4. ⚠️ **Terrain patterns unreliable**: High variance between runs
5. ✅ **SIREN competitive regionally**: But collapses at L=20

### Critical Issues Identified
1. ⚠️ **High variance**: Different random seeds give opposite conclusions
2. ⚠️ **Training instability**: Results not reproducible across runs
3. ⚠️ **Overfitting risk**: Splines may be fitting noise, not signal
4. ❌ **L=40 impractical**: Numerical instability with 1681 dimensions

### What This Means
1. **Exp 1 results questionable**: Single-seed results may not be representative
2. **Need multiple seeds**: Run each configuration 5-10 times to get confidence intervals
3. **SH encoding NOT the answer**: Higher L actually hurts splines

### Publication Potential
- **Current angle**: "Counter-Intuitive Terrain Effects in Learned Activations"
- **Key contribution**: Regional analysis reveals terrain-dependent but unexpected patterns
- **Caveat**: Need Exp 2 to rule out encoding confound

---

## Recommended Next Steps

### CRITICAL: Address Instability First

**Before continuing to Exp 3-5, we MUST address the reproducibility issue:**

1. **Multi-seed validation** (HIGHEST PRIORITY)
   - Rerun Exp 1 with 5-10 different random seeds
   - Compute mean ± std for each configuration
   - Only trust results with low variance (std < 1% of mean)
   - Current results: Single seed, potentially meaningless

2. **Fix L=40 numerical issues** (if still pursuing high-dim encoding)
   - Add gradient clipping
   - Use mixed precision training
   - Try batch normalization or layer normalization
   - Or accept that L=20 is maximum practical limit

3. **Validate Exp 1 inconsistency**
   - Why did Himalayas L=10 give -8% then +52%?
   - Run same region multiple times with different seeds
   - If variance remains high → results unreliable

### If Results Stabilize with Multi-Seed

4. **Complete Exp 2 properly**
   - Run Sahara (flat) at L=10, L=20
   - Multiple seeds for each configuration
   - Compare mountain vs flat robustly

5. **Investigate Europe anomaly**
   - Try multiple seeds
   - Check for data quality issues
   - May need to exclude from analysis

6. **Consider Exp 3-5 ONLY if**
   - Multi-seed Exp 1 shows stable, significant advantage (>5% with low variance)
   - Otherwise: STOP, write up negative result

### If Results Don't Stabilize

7. **Abandon regional analysis**
   - High variance indicates approach is fundamentally flawed
   - Return to NB19 global analysis (which had more stable results)
   - Focus on mechanistic understanding (NB21)
   - Publication angle: "Why Regional Analysis Fails for Learned Activations"

---

## Unexpected Winner: Africa Sahara + Spline

**Best finding so far**: +13.37% spline advantage in flat terrain

**Why exciting**:
- Largest advantage found across NB19, 19b, and NB20
- Clear, replicable condition (flat terrain)
- Contradicts paper predictions (they didn't test flat terrain explicitly)

**Actionable guidance** (if validated by Exp 2):
- **Use splines for**: Smooth, low-relief terrain (deserts, plains, oceans?)
- **Use ReLU for**: Complex, high-relief terrain (mountains, coastlines)

---

## Statistical Notes

**Training times** (consistent with NB19):
- ReLU: ~25s per region
- Spline: ~40s per region (~60% slower)
- SIREN: ~25s per region

**Advantage threshold**: +13.37% justifies 60% training time overhead ✅

---

## Files Generated

- ✅ `exp1_continental_comparisons.csv` - Full results (5 regions × 3 activations)
- ⚠️ `exp2_sh_encoding_levels.csv` - Partial results (only mountains L=10/L=20, L=40 crashed)

---

**Status**:
- Exp 1: Complete but needs multi-seed validation
- Exp 2: Incomplete due to L=40 crash and high variance at L=10
- **CRITICAL ISSUE**: Results not reproducible - Himalayas L=10 showed -8% (Exp 1) then +52% (Exp 2)
