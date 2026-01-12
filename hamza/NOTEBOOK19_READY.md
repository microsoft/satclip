# Notebook 19: Ready to Execute ✅

**Date**: 2026-01-12
**Status**: All systems go!

---

## Quick Summary

**Notebook 19 is fully prepared and ready to execute on Colab.**

### What It Does
Tests the **simplicity bias hypothesis** from Teney et al. (2024) CVPR paper to find "alpha" - where learned activations excel over ReLU.

### 5 Experiments Implemented
1. ⭐⭐⭐ **Regression vs Classification** - Does formulation matter?
2. ⭐⭐⭐ **High-Frequency Tasks** - Elevation (sharp) vs Population (smooth)
3. ⭐⭐ **Multi-Resolution Analysis** - Does finer resolution help splines?
4. ⭐⭐ **Function Complexity** - Total Variation measurement
5. ⭐ **Task Difficulty Scaling** - Smooth → Raw → Gradient

---

## Data Sources - All Validated ✅

| Dataset | Size | Status | Source |
|---------|------|--------|--------|
| **GPW Population** | 0 MB | ✅ In Drive | Already have |
| **ETOPO Elevation** | 60 MB | ✅ Validated | NOAA (direct download) |
| **Natural Earth Coastlines** | 3 MB | ✅ Validated | CDN (direct download) |
| **Temperature** (optional) | 1 MB | ✅ Validated | Figshare |

**Total download**: ~63 MB, 3-5 minutes
**All URLs tested**: HTTP 200 responses confirmed 2026-01-12

See [NB19_DATA_VALIDATION_SUMMARY.md](NB19_DATA_VALIDATION_SUMMARY.md) for details.

---

## Files Created

### Experimental Design
- **[NOTEBOOK19_SIMPLICITY_BIAS_TESTS.md](NOTEBOOK19_SIMPLICITY_BIAS_TESTS.md)** - Comprehensive experimental plan
  - Motivation from paper
  - 5 experiments with expected outcomes
  - Success criteria
  - Connection to literature

### Executable Notebook
- **[19_simplicity_bias_tests.ipynb](19_simplicity_bias_tests.ipynb)** - Ready to run on Colab
  - Data acquisition (auto-download)
  - Model definitions (Spline, ReLU, SIREN)
  - Training utilities
  - All 5 experiments implemented
  - Final summary and analysis

### Data Documentation
- **[NOTEBOOK19_DATA_SOURCES.md](NOTEBOOK19_DATA_SOURCES.md)** - Complete data documentation
  - All sources with validated URLs
  - Loading code templates
  - Fallback options

- **[NOTEBOOK19_DATA_QUICKSTART.py](NOTEBOOK19_DATA_QUICKSTART.py)** - Copy-paste acquisition script
  - Standalone Python script
  - All downloads in one place
  - Verification code included

- **[NB19_DATA_VALIDATION_SUMMARY.md](NB19_DATA_VALIDATION_SUMMARY.md)** - Validation results
  - HTTP response codes
  - File sizes
  - Download times
  - Fallback plans

---

## How to Execute

### Option 1: Run Full Notebook (Recommended)
1. Open [19_simplicity_bias_tests.ipynb](19_simplicity_bias_tests.ipynb) in Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (Runtime → Run all)
4. Wait ~5-6 hours for completion
5. Download results CSVs

### Option 2: Run Experiments Individually
1. Open notebook in Colab
2. Run setup cells (1-10)
3. Pick which experiments to run:
   - Exp 1: Cells 11-13 (~30 min)
   - Exp 2: Cells 14-16 (~2 hours)
   - Exp 3: Cells 17-19 (~1.5 hours)
   - Exp 4: Cells 20-22 (~30 min)
   - Exp 5: Cells 23-25 (~1 hour)

### What You'll Get
- `exp1_regression_vs_classification.csv` - Formulation comparison
- `exp2_high_frequency_tasks.csv` - Task frequency analysis
- `exp3_multi_resolution.csv` - Resolution scaling
- `exp4_complexity_measurement.csv` - Total Variation metrics
- `exp5_task_difficulty.csv` - Difficulty scaling

---

## Expected Outcomes

Based on Teney et al. (2024) predictions:

### Should Find:
✅ **Spline > ReLU** on regression (not classification)
✅ **Spline > ReLU** on elevation (high-frequency)
✅ **Spline advantage** increases with resolution
✅ **Complexity (TV)** correlates with performance
✅ **Spline advantage** increases with task difficulty

### What Success Looks Like:
If **ANY** of these hypotheses hold → **We found alpha!**

This means we've identified concrete scenarios where learned activations provide real advantages over ReLU.

---

## Connection to MOSAIKS Paper

The MOSAIKS repository validates our approach:
- ✅ They use GPW Population (we have it)
- ✅ They use elevation data (we have ETOPO 2022)
- ✅ They test multiple tasks with varying characteristics (smooth vs complex)
- ✅ Standard sample size: N=100,000 (we can match)

**Key insight**: Testing task diversity is the right approach.

---

## Technical Details

### Model Architecture
- **Input**: SH(L=10) features (121 dimensions)
- **Encoder**: 3×256 MLP
- **Activations**: Spline (k=15, relu init), ReLU, SIREN
- **Head**: 256→128→1 (regression) or 256→128→100 (classification)

### Training
- **Optimizer**: Adam, lr=1e-3
- **Epochs**: 80-100 (varies by experiment)
- **Batch size**: 256
- **Split**: 70/30 train/test with spatial blocking (5° grid)

### Computational Requirements
- **GPU**: T4 (free Colab tier)
- **RAM**: ~12 GB peak
- **Time**: ~5-6 hours total
- **Storage**: ~100 MB (CSVs + checkpoints)

---

## Troubleshooting

### If Data Download Fails:
1. **ETOPO elevation**: Use Open-Elevation API (fallback in QUICKSTART)
2. **Coastlines**: Try alternative Natural Earth URL in data docs
3. **Temperature**: Optional, can skip

### If Training OOMs:
1. Reduce `n_samples` in experiments
2. Reduce `batch_size` to 128
3. Use Colab Pro for more RAM

### If Results Unexpected:
1. Check data loaded correctly (verify shapes)
2. Check normalization (elevation needs special handling)
3. Try multiple random seeds (robustness test)

---

## After Execution

### Immediate Next Steps:
1. **Review CSV results** - Check if hypotheses confirmed
2. **Create ANALYSIS_NOTEBOOK19.md** - Document findings
3. **Update README.md** - Add key conclusions
4. **Update EXPERIMENTS_TRACKER.md** - Mark NB19 complete

### Follow-up Notebooks:
- **NB20**: Visualize learned activation shapes
- **NB21**: Test robustness (multiple seeds, error bars)
- **NB22**: Architecture interaction (depth/width sweeps)
- **NB23**: Raw + Learned deep dive

---

## Key Questions to Answer

After running NB19, you should be able to answer:

1. **Primary**: **When do learned activations beat ReLU?**
   - Regression vs classification?
   - High-frequency vs low-frequency?
   - Fine vs coarse resolution?

2. **Secondary**: **What's the magnitude of improvement?**
   - Small (<1%) on smooth tasks?
   - Large (>5%+) on high-frequency tasks?

3. **Mechanistic**: **Does complexity correlate with performance?**
   - As paper predicts for regression?
   - Different for classification?

4. **Practical**: **What's the resolution threshold?**
   - Below X km → ReLU wins?
   - Above X km → Spline wins?

---

## References

### Papers
- **Teney et al. (2024)** "Do We Always Need the Simplicity Bias?" CVPR
  - [SIMPLICITY_PAPER.md](SIMPLICITY_PAPER.md)

### Our Previous Work
- **NB18**: Spline deep dive (optimal config: k=15, relu init)
- **NB16**: Core 2×2 comparison (SH+ReLU: +2.93% vs SIREN)
- **NB17**: RFF+SH diagnostic (frequency interference)
- **NB15**: Multi-resolution baseline

---

## Contact

**Questions?** See:
- [NOTEBOOK19_SIMPLICITY_BIAS_TESTS.md](NOTEBOOK19_SIMPLICITY_BIAS_TESTS.md) - Full experimental design
- [NB19_DATA_VALIDATION_SUMMARY.md](NB19_DATA_VALIDATION_SUMMARY.md) - Data validation details
- [README.md](README.md) - Project overview

---

**Ready to find alpha! 🎯**

Execute the notebook and let's discover where learned activations truly excel.
