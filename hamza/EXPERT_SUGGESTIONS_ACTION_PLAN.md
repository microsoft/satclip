# Expert Suggestions: Analysis & Action Plan

**Date**: 2026-01-12
**Status**: NB21/21b running, NB21c ready to launch

---

## Executive Summary

**Expert Consensus (4/4 experts)**: The critical missing piece is **raw coordinate multi-seed validation**.

**What's Already Addressed**:
- ✅ Multi-seed evaluation (NB21/21b with 10 seeds)
- ✅ Confidence intervals & statistical rigor
- ✅ Regional analysis with variance
- ✅ Sample size sensitivity (5K to 50K)
- ✅ Convergence testing (200 epochs)

**What's Still Missing**:
- ❌ Raw coordinates with same multi-seed rigor
- ❌ Performance/parameter analysis
- ❌ Sharp boundary tasks (coastlines)
- ❌ Validation split infrastructure

---

## What the Experts Said

### Expert 1: "SH Pre-Encoding Likely Hides Effects"

**Key quote**:
> "The simplicity paper's strongest effects appear without hand-crafted frequency encodings. SH(L=10) may pre-smooth signals and remove the need for learned activations."

**Suggested task**:
> "Replicate NB19 experiments using raw coordinates only"

**Priority**: 🥇 HIGHEST

---

### Expert 2: "Raw + Learned Is the Alternative Path"

**Key quote**:
> "If you want a broader learned-activation story, keep learned activations with raw coords (where RFF/Spline can compete)."

**Focus areas**:
1. Raw + learned (RFF or spline) as alternative to SH
2. Multi-seed robustness (NOW DONE in NB21/21b)
3. High-frequency tasks to stress learned activations

**Priority**: 🥇 HIGHEST (raw coords), 🥈 MEDIUM (high-freq tasks)

---

### Expert 3: "Performance Per Parameter, Not Training Time"

**Key quote**:
> "The sharper question is performance per parameter (and per dataset/scale) rather than training time—your own notes already show learned activations typically cost more time without consistent gains."

**Suggested tasks**:
1. Match model capacity across input encodings
2. Add high-frequency, boundary-heavy tasks
3. Stress-test spatial leakage controls

**Priority**: 🥈 MEDIUM (perf/param analysis), 🥉 LOW (spatial blocking)

---

### Expert 4: "Reproducibility First, Then Everything Else"

**Key quote**:
> "Single-seed results swing from −8% to +52% on the same region/encoding, which makes any conclusion unreliable."

**Immediate focus**:
1. Make results reproducible (NOW DONE in NB21/21b)
2. Add raw-coordinate baselines for high-frequency tasks
3. Add validation-based early stopping

**Priority**: ✅ DONE (multi-seed), 🥇 HIGHEST (raw coords), 🏅 MEDIUM (validation split)

---

## Current State Analysis

### What's Deprecated (Already Addressed by NB21/21b)

✅ **"Multi-seed evaluation"**
- NB21/21b: 10 seeds per configuration
- Mean±std, 95% CI, paired t-tests
- CV analysis for stability

✅ **"Regional analysis with variance"**
- NB21 Exp 3: Himalayas/Sahara with 10K/20K samples
- 10 seeds per region/size combination

✅ **"Sample size sensitivity"**
- NB21/21b Exp 2: 5K, 10K, 20K, 50K samples
- Variance vs N analysis

✅ **"Convergence testing"**
- NB21/21b Exp 4: 200 epochs with R² tracking
- Tests if 100 epochs sufficient

### What's Still Missing

❌ **Raw coordinate multi-seed validation**
- Have single-seed results from NB14-16
- But no statistical rigor, can't trust small differences
- **ALL 4 EXPERTS** identified this as critical gap

❌ **Performance/parameter analysis**
- Small spline gains may not justify parameter cost
- Need: R² per 1K parameters, efficiency metrics
- Easy to add once NB21/21b results available

❌ **Sharp boundary task**
- Coastline distance (step-like discontinuities)
- Tests simplicity paper's "sharp transition" hypothesis
- Mentioned by 3/4 experts

❌ **Validation split infrastructure**
- Current train/test split risks test leakage
- Need proper train/val/test for hyperparameter selection
- Mentioned by 2/4 experts

---

## Priority Ranking

### 🥇 Priority 1: NB21c - Raw Coordinate Multi-Seed Validation

**Why highest priority**:
- Critical control experiment all experts converged on
- Directly tests if SH masks learned activation benefits
- Uses existing infrastructure (copy NB21, change input_type)
- Fast execution (~7 hours)

**What it tests**:
1. Can splines beat ReLU without SH? (Raw+Spline vs Raw+ReLU)
2. Does SH add value? (SH+ReLU vs Raw+ReLU)
3. Does SH mask spline gains? (Compare spline advantages with/without SH)

**Expected outcomes**:
- **If Raw+Spline wins**: SH was masking benefits → investigate why
- **If Raw+ReLU wins**: Confirms tasks don't benefit from learned acts
- **If SH >> Raw**: Justifies SH parameter cost

**Status**: ✅ Notebook created, ready to run

**Addresses**: Expert 1 Issue 3, Expert 2 "raw+learned", Expert 3 Issue 1, Expert 4 Issue 3

---

### 🥈 Priority 2: Performance/Parameter Analysis

**Why second priority**:
- Can be done immediately once NB21/21b complete
- Answers "is small spline gain worth parameter cost?"
- Uses existing data, no new experiments

**What to calculate**:
```python
# For each config:
- Total parameters
- R² per 1K parameters
- Training time per epoch
- R² improvement per added parameter (vs ReLU)

# Visualizations:
1. Pareto frontier: R² vs Parameters
2. Efficiency: (R² gain) / (param increase) / (time cost)
3. Statistical: Is spline gain > parameter cost?
```

**Example analysis**:
```
SH+ReLU:   0.900 R², 500K params → 1.80 R² / 1K params
SH+Spline: 0.903 R², 550K params → 1.64 R² / 1K params

Spline adds 50K params for +0.3% R² gain
→ Cost-benefit: Not justified
```

**Status**: Analysis code ready, waiting for NB21/21b results

**Addresses**: Expert 1 "perf/parameter", Expert 2 "performance per parameter", Expert 3 Issue 3

---

### 🥉 Priority 3: Sharp Boundary Task - Coastline Distance

**Why third priority**:
- Tests simplicity paper's "sharp transition" hypothesis
- Mentioned by 3/4 experts
- Requires new data prep (coastline vectors)

**Task design**:
```python
Task: Distance to nearest coastline (meters, log-transformed)
Data: Natural Earth coastline vectors + ETOPO DEM
Properties:
  - Step-like discontinuities at land/water boundary
  - High-frequency content near coasts
  - Global coverage for spatial blocking

Hypothesis: Splines should excel IF simplicity paper is right
Test with: Multi-seed (10 runs) from the start
```

**Expected outcome**:
- Strong test of "learned activations for high-frequency content"
- If splines don't win here, they won't win anywhere

**Status**: Not started, ~5 hours work (data prep + training)

**Addresses**: Expert 2 "coastline distance", Expert 3 Issue 4, Expert 4 Issue 6

---

### 🏅 Priority 4: Validation Split Infrastructure

**Why fourth priority**:
- Improves rigor of all future experiments
- But doesn't change conclusions from past work
- Mentioned by 2/4 experts

**Implementation**:
```python
def sample_with_validation(data, lons, lats,
                          train_ratio=0.6,
                          val_ratio=0.2,
                          test_ratio=0.2):
    # Assign each spatial block to train/val/test
    # Use validation for:
    #   - Early stopping
    #   - Model selection (which activation?)
    #   - Hyperparameter tuning (LR, etc.)
    # Only look at test set for final reporting
```

**Benefits**:
- Prevents test leakage in model selection
- Enables proper early stopping
- Standard ML best practice

**Status**: Not implemented

**Addresses**: Expert 2 Issue 3, Expert 3 Issue 5, Expert 4 Issue 4

---

### 📊 Priority 5: Cross-Task Synthesis (After NB21/21b/21c)

**Why fifth priority**:
- Requires all three notebooks to complete first
- But is critical for final conclusions

**Analysis questions**:
1. **Task-dependence**: Elevation vs population patterns
2. **Encoding-dependence**: Raw vs SH patterns
3. **Resolution effect**: Does native resolution matter? (NB21b Exp 3)
4. **Boundary effect**: Do urban boundaries help splines? (NB21b Exp 4)
5. **Convergence**: Is 100 epochs enough? (All notebooks Exp 5)
6. **Spatial blocking**: Does grid size matter? (NB21 Exp 5)

**Output**:
- Comprehensive comparison table
- Decision tree: "When to use which activation + encoding"
- Either strong positive OR strong negative result (both publishable)

**Status**: Waiting for all notebooks to complete

---

## What We're NOT Doing (and Why)

### ❌ Meta-Learning Activations

**Expert mention**: All 4 experts mentioned bi-level/episodic meta-learning

**Why skip**:
- Complex infrastructure change
- Low chance of deployment in practice
- Would only pursue if NB21c shows promising raw+spline results
- Simplicity paper's setup, not universally applicable

**Decision**: Skip unless raw+spline shows >3% reproducible advantage

---

### ❌ SRTM 30m High-Resolution Elevation

**Expert mention**: Mentioned as "high-frequency terrain" data

**Why skip for now**:
- ETOPO 60s (2km) sufficient for testing hypothesis
- SRTM 30m is 400× larger dataset
- Expensive to process and train on
- Would test same hypothesis as ETOPO

**Decision**: Current ETOPO sufficient, can add SRTM later if needed

---

### ❌ OSM Building Footprints

**Expert mention**: "Sharp urban boundaries" task

**Why skip for now**:
- Very task-specific (urban mapping)
- Complex preprocessing (vector to raster)
- Coastline distance tests same hypothesis (step functions)

**Decision**: Coastline distance is cleaner test of sharp boundaries

---

### ❌ Additional Spatial Blocking Tests

**Expert mention**: "Stress-test spatial leakage controls"

**Why low priority**:
- NB21 Exp 5 already tests 1°, 2°, 5°, 10° grids
- Current 5° blocking seems adequate (based on NB19/20)
- Would only matter if results were on borderline

**Decision**: Wait for NB21 Exp 5 results before additional tests

---

## Execution Timeline

### Immediate (Now)

1. ✅ Review EXPERT_SUGGESTIONS.md
2. ✅ Create NB21c notebook structure
3. ✅ Create NOTEBOOK21C_DESIGN.md
4. ⏳ Launch NB21c when GPU available

**Estimated time**: NB21c ~7 hours

---

### After NB21/21b Complete (~12 hours from launch)

1. Analyze NB21/21b results
2. Perform Priority 2 (Performance/Parameter Analysis)
3. Perform Priority 5 (Cross-Task Synthesis)
4. If results promising → Launch NB21c
5. If results clear negative → Write up conclusions

**Estimated time**: 2-3 hours analysis

---

### After NB21c Complete (~7 hours from launch)

1. **Critical comparison**: Raw vs SH encoding
   - Does SH add value? (SH+ReLU vs Raw+ReLU)
   - Does SH mask spline gains? (Compare advantages)
   - Can splines win without SH? (Raw+Spline vs Raw+ReLU)

2. **Decision point**:
   - If Raw+Spline wins → Investigate mechanistically
   - If Raw+ReLU wins → Write up "simplicity sufficient"
   - If SH >> Raw → Confirm SH is essential

3. **Consider Priority 3** (Coastline task):
   - Only if splines show promise in NB21c
   - Would be strongest test of high-freq hypothesis

**Estimated time**: 1 hour comparison + decision

---

### Total Timeline

**Critical path**:
- NB21/21b: ~12 hours (running now)
- NB21c: ~7 hours (ready to launch)
- Analysis: ~3 hours
- **Total**: ~22 hours to complete core validation

**Optional follow-ups**:
- Coastline task: ~5 hours
- Validation split: ~2 hours implementation
- **Total with optional**: ~29 hours

---

## Expected Publication Outcomes

### Scenario A: Raw+Spline Wins (Best Case)

**Finding**: "SH Pre-Encoding Masks Learned Activation Benefits"

**Key results**:
- Raw+Spline advantage: +3% (significant, reproducible)
- SH+Spline advantage: +0.3% (marginal)
- SH masks 10× larger benefit

**Publication angle**:
- When to use raw+spline vs SH+ReLU
- Why frequency encodings can hurt learned activations
- Guidance for practitioners

**Impact**: HIGH - Challenges common practice of always using encodings

---

### Scenario B: SH+Acts >> Raw+Acts (Good)

**Finding**: "Why Geographic Tasks Need Frequency Encodings"

**Key results**:
- SH+ReLU: +20% over Raw+ReLU
- SH+Spline: Similar advantage
- Raw coords fundamentally unstable (CV > 50%)

**Publication angle**:
- Empirical justification for SH encoding
- Performance/parameter tradeoff analysis
- When SH overhead is justified

**Impact**: MEDIUM - Validates existing practice

---

### Scenario C: Raw+ReLU Still Wins (Negative Result, Still Good)

**Finding**: "When Simplicity Bias Is Sufficient: Geographic Prediction"

**Key results**:
- Raw+ReLU ≥ Raw+Spline (reproducible)
- SH+ReLU ≥ SH+Spline (reproducible)
- Learned activations don't help, even without SH

**Publication angle**:
- When learned activations fail
- Task properties that preclude benefits
- Cost-benefit of architectural complexity

**Impact**: MEDIUM - Important negative result, high rigor

---

### Scenario D: High Variance Everywhere (Weak but Honest)

**Finding**: "Reproducibility Challenges in Learned Activation Evaluation"

**Key results**:
- CV > 50% even with 10 seeds
- Results unstable across regions, samples, encodings
- No clear winner in any configuration

**Publication angle**:
- Methodological challenges
- Need for stronger evaluation protocols
- When small differences are noise

**Impact**: LOW - But honest, rigorous science

---

## Bottom Line

### What to Run Next

**Immediate**: Launch NB21c (raw coordinates multi-seed) when GPU available

**After NB21/21b complete**:
1. Performance/parameter analysis
2. Cross-task synthesis
3. Compare Raw vs SH results from NB21c

**Optional follow-ups**:
- Coastline task (if splines show promise)
- Validation split (for future rigor)

### What Makes This High-Alpha

1. **Direct test of mechanistic hypothesis** all experts converged on
2. **Uses existing infrastructure** (copy NB21, change input_type)
3. **Fast execution** (~7 hours)
4. **Clear outcomes** (win/lose/high-variance, all publishable)
5. **Answers the key question**: Does SH mask learned activation benefits?

### Expected Timeline

- **Now**: NB21c ready to launch
- **+12 hours**: NB21/21b results available
- **+19 hours**: NB21c results available
- **+22 hours**: Complete analysis, clear conclusion

**Within 1 day of GPU time**, we'll know definitively whether SH encoding masks learned activation benefits.

This is the **critical missing control experiment** needed before making any claims.
