# Notebook 21: Reproducibility Validation Design

**Created**: 2026-01-12
**Purpose**: Establish reproducible baselines before making any claims about learned activations

---

## Problem Statement

NB20 revealed **critical reproducibility issues**:

### Same Region, Wildly Different Results

**Asia Himalayas (L=10)**:
- Run 1: Spline -8.22% (ReLU wins)
- Run 2: Spline -8.83% (ReLU wins)
- Run 3: Spline **+52.0%** (Spline wins!)

**Africa Sahara (L=10)**:
- Run 1: Spline +13.37% (Spline wins)
- Run 2: Spline +1.96% (barely wins)

**Variance range**: 60 percentage points for same configuration

**Root cause**: Single-seed runs with small samples (5K) and coarse spatial blocking (5°)

---

## NB21 Solution: Multi-Seed Validation

### Core Principle

**No claims without statistical validation:**
- 10 runs per configuration (seeds 42-51)
- Report mean ± std, 95% CI
- Check coefficient of variation (CV)
- Paired t-tests for comparisons

### Success Criteria

Results considered **stable** if:
1. CV < 20% (std < 20% of mean)
2. 95% CI excludes zero (if claiming advantage)
3. Consistent across sample sizes ≥20K

Results considered **unstable** if:
- CV > 50%
- CI includes wide range of effects
- Mean changes direction with sample size

---

## Experimental Design

### Experiment 1: Global Multi-Seed (NB19 Validation)

**Config**: 15K samples, global, L=10, 100 epochs

**Question**: Is NB19's +0.36% spline advantage reproducible?

**Expected outcome**:
- **If NB19 was lucky**: Mean ~0%, high variance
- **If NB19 was representative**: Mean ~0.3%, low variance
- **Allows**: Validation of baseline before regional analysis

**Computational cost**: 10 seeds × 3 acts × 90s = ~45 min

---

### Experiment 2: Sample Size Sensitivity

**Configs**: 5K, 10K, 20K, 50K samples (global)

**Question**: At what sample size do results stabilize?

**Expected pattern**:
- 5K: High variance (CV > 50%)
- 10K: Moderate variance (CV 20-50%)
- 20K: Low variance (CV < 20%)
- 50K: Minimal variance (CV < 10%)

**Key insight**: Determines minimum N for reliable regional analysis

**Computational cost**: 4 sizes × 10 seeds × 2 acts × 60-120s = ~4 hours

---

### Experiment 3: Regional Multi-Seed (Larger Samples)

**Configs**: Himalayas/Sahara, 10K/20K samples, 10 seeds

**Question**: Do terrain patterns stabilize with more data?

**Possible outcomes**:

**Outcome A: Stable terrain effect**
- Himalayas: Spline -5% ± 2% (ReLU wins consistently)
- Sahara: Spline +8% ± 2% (Spline wins consistently)
- **Interpretation**: Real terrain effect, proceed with NB20 Exp 3-5
- **Publication**: "Terrain-Dependent Learned Activation Performance"

**Outcome B: No stable pattern**
- Himalayas: Spline +2% ± 15% (wide CI includes zero)
- Sahara: Spline -1% ± 12% (inconsistent)
- **Interpretation**: NB20 results were noise
- **Publication**: "High Variance in Regional Activation Analysis"

**Outcome C: Both favor ReLU**
- Himalayas: Spline -3% ± 2%
- Sahara: Spline -2% ± 2%
- **Interpretation**: No terrain effect, ReLU always wins
- **Publication**: "SH + ReLU Optimal for Geographic Prediction"

**Computational cost**: 2 regions × 2 sizes × 10 seeds × 2 acts × 100s = ~3 hours

---

### Experiment 4: Extended Training (Convergence)

**Config**: 200 epochs (vs 100), track R² every epoch, 5 seeds

**Question**: Were models undertrained in NB19/20?

**Possible outcomes**:

**If models plateaued by epoch 100**:
- No improvement from 100 → 200
- **Interpretation**: 100 epochs sufficient, variance is intrinsic
- **Action**: Continue with 100 epochs

**If models still improving**:
- Significant R² gain from 100 → 200
- **Interpretation**: Undertraining explains variance
- **Action**: Increase epochs to 200-500, rerun all experiments

**Also reveals**: Which activation converges faster

**Computational cost**: 5 seeds × 2 acts × 180s = ~45 min

---

### Experiment 5: Spatial Blocking Sensitivity

**Configs**: 1°, 2°, 5°, 10° grid sizes, 5 seeds

**Question**: How much does blocking strategy affect results?

**Theory**: Smaller blocks → more test cells → better spatial coverage → lower variance

**Expected pattern**:
- 10° (648 blocks): Highest variance
- 5° (2592 blocks): Moderate variance (NB19/20 baseline)
- 2° (16,200 blocks): Lower variance
- 1° (64,800 blocks): Lowest variance

**Key insight**:
- If variance drops with smaller grids → blocking was the issue
- If variance constant → intrinsic model/sampling variability

**Computational cost**: 4 grids × 5 seeds × 2 acts × 90s = ~90 min

---

## Total Computational Budget

**Total time**: ~10 hours on Colab T4 GPU

**Breakdown**:
- Exp 1: 45 min
- Exp 2: 4 hours
- Exp 3: 3 hours
- Exp 4: 45 min
- Exp 5: 90 min

**Acceptable**: This is a one-time validation, critical for scientific rigor

---

## Output Strategy

### All results printed in notebook

**For each configuration**:
```
Mean ± Std:  0.9030 ± 0.0120
Range:       [0.8850, 0.9180]
95% CI:      [0.8940, 0.9120]
CV:          1.33%
N:           10
```

**For comparisons**:
```
Spline Advantage (%):
  Mean ± Std:  +2.34% ± 1.20%
  Range:       [+0.50%, +4.10%]
  95% CI:      [+1.52%, +3.16%]

Paired t-test: t=3.456, p=0.0034
  ✅ SIGNIFICANT: Spline wins (p < 0.05)
  🎯 PRACTICAL SIGNIFICANCE: 95% CI excludes zero
```

### No reliance on CSVs

- All key info in notebook output
- User can read results directly from executed cells
- Optional Google Drive save for persistence

---

## Decision Tree After NB21

### If Exp 1 shows stable global results (CV < 20%)

**AND Exp 3 shows stable terrain effect**:
- ✅ Proceed with NB20 Exp 3-5 using 20K+ samples
- Use multi-seed (5-10 runs) for all configurations
- Publication: "When and Where Learned Activations Excel"

**AND Exp 3 shows no stable pattern**:
- ⚠️ Regional analysis unreliable, stick to global
- Publication: "SH + ReLU Optimal for Global Geographic Prediction"

### If Exp 1 shows high variance (CV > 50%)

**Even at global scale**:
- ❌ Fundamental reproducibility issue
- Focus on understanding WHY (NB22: Mechanistic Analysis)
- Possible causes:
  - Network initialization sensitivity
  - Optimization landscape issues
  - Task-specific instability
- Publication: "Challenges in Evaluating Learned Activations"

### If Exp 2 shows variance decreases with N

**Find minimum stable N**:
- Use that N for all future experiments
- May need 50K+ samples (acceptable with compute budget)

### If Exp 4 shows undertraining

**Increase epochs**:
- Rerun critical experiments with 200-500 epochs
- Check if variance was due to incomplete convergence

### If Exp 5 shows blocking matters

**Use finer grids**:
- Switch to 1° or 2° blocks for all experiments
- Rerun regional analysis with better spatial CV

---

## Key Differences from NB19/20

### NB19/20 Approach (Problematic)

- Single seed per configuration
- Small samples (5K regional, 15K global)
- Coarse blocking (5° grid)
- 100 epochs (may be insufficient)
- **Result**: Unreproducible, contradictory findings

### NB21 Approach (Rigorous)

- 10 seeds per configuration (5 for convergence/blocking)
- Vary sample sizes (5K to 50K)
- Test blocking sensitivity
- Test convergence
- Statistical validation (t-tests, CIs, CV)
- **Expected**: Clear signal vs noise separation

---

## What We'll Learn

### Best case: Stable, significant effect

- Spline advantage reproducible
- Clear terrain or scale pattern
- Practical guidance on when to use splines
- Strong publication

### Middle case: Stable, small effect

- Spline ~0.5% better (significant but not practical)
- Consistent across seeds but not worth compute cost
- Publication on cost-benefit analysis

### Worst case: High variance

- Even with multi-seed, no clear winner
- Task/domain-specific instability
- Publication on reproducibility challenges
- Still valuable negative result

---

## Statistical Rigor

### Metrics Reported

1. **Point estimates**: Mean (primary), Median (robust)
2. **Spread**: Std, IQR, Min/Max
3. **Uncertainty**: 95% CI, Std Error
4. **Variability**: CV (std/mean × 100%)
5. **Significance**: Paired t-test (same seeds), p-values

### Interpretation Guidelines

**Coefficient of Variation (CV)**:
- CV < 10%: Very stable
- CV 10-20%: Acceptable
- CV 20-50%: Moderate variance, interpret with caution
- CV > 50%: High variance, unreliable

**95% Confidence Interval**:
- CI excludes zero + mean > 1%: Practical significance
- CI excludes zero + mean < 1%: Statistical but not practical
- CI includes zero: Inconclusive

**Paired t-test**:
- p < 0.001: Very strong evidence
- p < 0.01: Strong evidence
- p < 0.05: Significant
- p ≥ 0.05: Not significant

---

## Expected Timeline

1. **Run NB21**: ~10 hours on Colab (can split into sessions)
2. **Analyze results**: 1-2 hours (mostly automated in notebook)
3. **Decision point**: Proceed, pivot, or stop
4. **Next notebook** (depends on results):
   - If stable: NB20 continuation with larger samples
   - If unstable: NB22 mechanistic analysis
   - If converged: Final publication prep

---

## Success Metrics

NB21 is **successful** if it provides:

1. ✅ **Clear variance estimates** for all configurations
2. ✅ **Stable baseline** for global predictions (Exp 1)
3. ✅ **Sample size guidance** for future experiments (Exp 2)
4. ✅ **Validated terrain effects** OR clear evidence they don't exist (Exp 3)
5. ✅ **Training protocol validation** (Exp 4 & 5)

**Even if results are "negative" (no spline advantage), NB21 succeeds if it gives us confidence in the findings.**

---

## Notes

- All output printed in notebook (no hidden CSVs)
- Can run in parallel if multiple GPUs available
- Can split into multiple Colab sessions (save intermediate DFs)
- Google Drive backup optional but recommended
- Results form foundation for all future experiments

**Ready to run when user confirms approach.**
