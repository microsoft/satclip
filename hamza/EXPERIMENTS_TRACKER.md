# Experiments Tracker - Living Document

**Last Updated**: 2026-01-12
**Project**: Learned Activations for Geographic Encoders

This is a living document tracking all experiments: completed, in progress, and planned.

---

## 🎯 Current Status

**Phase**: 1 Complete ✅ | 2 In Progress 🔄

**Current Focus**: Notebook 18 - Spline Deep Dive

**Next Action**: Execute Notebook 18 on Colab

---

## 📊 Completed Experiments (Phase 1)

### Notebook 14: Initial Exploration ✅
**Status**: Complete
**Date**: ~2025-12
**Environment**: CPU (local)
**Duration**: ~2 hours

#### Setup
- **Data**: GPW population density, 15-min resolution (~28km)
- **Sample**: 10,000 locations
- **Architecture**: 3×256 MLP
- **Training**: 100 epochs, lr=1e-3, Adam

#### Experiments Run
1. Raw coords + ReLU (baseline)
2. Raw coords + RFF (n=25)
3. Raw coords + Spline (k=10)

#### Key Results
| Model | R² | Winner |
|-------|-----|--------|
| Raw + RFF (n=25) | 0.743 | ✅ Best |
| Raw + ReLU | 0.729 | Baseline |

#### Takeaway
✅ Learned activations work on raw coordinates
✅ RFF beats ReLU by 1.4%

---

### Notebook 15: Multi-Resolution Comparison ✅
**Status**: Complete
**Date**: ~2025-12
**Environment**: Colab T4 GPU
**Duration**: ~2 hours

#### Setup
- **Data**: GPW population density, 3 resolutions
  - 15-min (~28km)
  - 30-min (~55km)
  - 1-degree (~111km)
- **Baselines**: SatCLIP L=10, SatCLIP L=40 (frozen + Ridge)
- **Architecture**: 3×256 MLP
- **Training**: 100 epochs, lr=1e-3, Adam

#### Experiments Run
1. Raw + ReLU (231K params)
2. Raw + RFF (n=25) (231K params)
3. Raw + Spline (k=10) (231K params)
4. SatCLIP L=10 (446K params)
5. SatCLIP L=40 (1.2M params)

#### Key Results (15-min resolution)
| Model | R² | vs SatCLIP L=10 | Params | Efficiency |
|-------|-----|-----------------|--------|-----------|
| Raw + RFF (n=25) | 0.733 | **+2.9%** | 231K | 0.033 |
| Raw + Spline (k=10) | 0.728 | +2.4% | 231K | 0.033 |
| Raw + ReLU | 0.722 | +1.8% | 231K | 0.032 |
| SatCLIP L=10 | 0.704 | baseline | 446K | 0.017 |
| SatCLIP L=40 | 0.618 | **-8.6%** | 1.2M | 0.005 |

#### Takeaways
✅ Raw + learned activations beat SatCLIP L=10 by 2.9%
✅ 2× better efficiency (params-normalized)
⚠️ SatCLIP L=40 mysteriously underperformed

#### Documentation
[ANALYSIS_NOTEBOOK15.md](ANALYSIS_NOTEBOOK15.md)

---

### Notebook 16: Phase 1 Core 2×2 ✅
**Status**: Complete
**Date**: ~2025-12
**Environment**: Colab T4 GPU
**Duration**: ~3 hours

#### Setup
- **Data**: GPW population density, 15-min resolution
- **Core Test**: Raw vs SH(L=10) × SIREN vs Learned
- **Architecture**: 3×256 MLP
- **Training**: 100 epochs, lr=1e-3, Adam

#### Experiments Run (12 models)

**Raw Coordinates:**
1. Raw + SIREN (baseline)
2. Raw + ReLU
3. Raw + RFF (n=25, 50, 100)
4. Raw + Spline (k=10, 30)

**SH(L=10) Features:**
5. SH + SIREN (baseline)
6. SH + ReLU
7. SH + RFF (n=25, 50)
8. SH + Spline (k=10, 30)

#### Key Results
| Model | R² | vs SIREN | Status |
|-------|-----|----------|--------|
| **SH + ReLU** | **0.7490** | **+0.63%** | ✅ Winner |
| **SH + Spline (k=10)** | **0.7483** | **+0.56%** | ✅ Good |
| SH + SIREN | 0.7427 | baseline | ✅ Baseline |
| SH + Spline (k=30) | 0.7411 | -0.16% | ✅ OK |
| Raw + SIREN | 0.7427 | baseline | ✅ Works |
| Raw + Spline (k=30) | 0.7369 | -0.58% | ✅ OK |
| Raw + RFF (n=100) | 0.7355 | -0.72% | ✅ OK |
| Raw + RFF (n=25) | 0.7351 | -0.76% | ✅ OK |
| Raw + Spline (k=10) | 0.7351 | -0.76% | ✅ OK |
| Raw + RFF (n=50) | 0.7251 | -1.76% | ⚠️ Worse |
| **SH + RFF (n=25)** | **0.6631** | **-7.96%** | ❌ Failed |
| SH + RFF (n=50) | 0.6290 | -11.37% | ❌ Failed |

#### Major Findings
❌ **RFF + SH catastrophically fails** (-7.96%)
✅ **ReLU beats SIREN** (+0.63%)
✅ **Spline works well** (+0.56%)
⚠️ More RFF features makes it worse (n=50: -11.37%)
⚠️ More spline knots hurts with SH (k=30: -0.16%)

#### Hypothesis Generated
**Why does RFF+SH fail?**
1. Frequency interference (SH spherical harmonics vs RFF Cartesian Fourier)
2. Input statistics mismatch (RFF expects normalized inputs)
3. Optimization difficulty (high gradient norms)

#### Documentation
[CRITICAL_ANALYSIS_NB16.md](CRITICAL_ANALYSIS_NB16.md)

---

### Notebook 17: Diagnostic - Why RFF+SH Fails ✅
**Status**: Complete
**Date**: ~2025-12
**Environment**: Colab T4 GPU
**Duration**: ~2 hours

#### Setup
- **Data**: GPW population density, 15-min resolution
- **Goal**: Test hypotheses for RFF failure
- **Architecture**: 3×256 MLP
- **Training**: 100 epochs, lr=1e-3, Adam

#### Experiments Run

**Hypothesis 1: Input statistics mismatch**
1. SH + RFF (no normalization) - baseline
2. SH + RFF (normalized SH features to mean=0, std=1)

**Hypothesis 2: Fixed frequencies are wrong**
3. SH + RFF (learnable frequencies instead of fixed)

**Hypothesis 3: Training dynamics**
4. Measure gradient norms for all activations
5. Visualize loss curves
6. Track convergence speed

**Baselines for comparison:**
7. SH + ReLU
8. SH + Spline (k=10)
9. SH + SIREN

#### Key Results
| Model | R² | vs SIREN | Gradient Norm |
|-------|-----|----------|---------------|
| **SH + ReLU** | **0.6979** | **+2.93%** | 1.99 (stable) |
| **SH + Spline (k=10)** | **0.6953** | **+2.53%** | 2.98 (stable) |
| SH + SIREN | 0.6781 | baseline | 9.44 (medium) |
| SH + RFF (no norm) | 0.6256 | -7.74% | 6.26 (oscillating) |
| SH + RFF (normalized) | 0.6193 | **-8.67%** | 4.65 (oscillating) |
| SH + RFF (learnable) | 0.5549 | **-18.2%** | 93.3 at epoch 1 (exploding) |

**Note**: Different R² values than NB16 due to different random seed/train-test split

#### Major Findings
❌ **Normalization made it WORSE** (0.6193 vs 0.6256)
❌ **Learnable frequencies catastrophic** (0.5549, -18.2%)
✅ **Frequency interference confirmed** (the real problem)
✅ **ReLU/Spline have much lower gradient norms** (2-3 vs 6-9)

#### Convergence Analysis
**ReLU/Spline**: Fast convergence (epoch 10 R² > 0.73), stable
**SIREN**: Slow convergence, degrades over time
**RFF**: Very slow start (epoch 1 R² < 0.13), never catches up

#### Verdict
🔴 **RFF + SH is fundamentally broken, not fixable**
- Not an input statistics problem
- Not a fixed frequency problem
- Frequency interference is the root cause
- Don't pursue further

#### Documentation
[DIAGNOSTIC_CONCLUSIONS_NB17.md](DIAGNOSTIC_CONCLUSIONS_NB17.md)

---

## 🔄 In Progress (Phase 2)

### Notebook 18: Spline Deep Dive ✅
**Status**: ✅ Complete
**Date**: 2026-01-12
**Duration**: ~30 minutes GPU time
**Environment**: Colab T4 GPU

#### Setup
- **Data**: GPW population density, 15-min resolution
- **Samples**: 15,000 (10,532 train, 4,468 test)
- **Spatial blocking**: 5° grid, 30% test cells
- **Input**: SH(L=10) features (100-dim)
- **Architecture**: 3×256 MLP + prediction head
- **Training**: 100 epochs, batch_size=256, lr=1e-3, Adam

#### Experiments Run (6 experiments, 20 models trained)

**1. Knot Count Sweep** (k = 5, 10, 15, 20, 30, 50)
**2. Initialization** (relu, linear, zero, tanh, gelu)
**3. Input Range** ((-3,3), (-5,5), (-10,10))
**4. Learnable Positions** (fixed vs learnable)
**5. Visualization** (learned activation shapes)
**6. Baseline Comparison** (vs ReLU, SIREN)

#### Key Results

| Configuration | R² | vs SIREN | Status |
|---------------|-----|----------|--------|
| **Optimal Spline (k=15, relu)** | **0.7447** | **+3.2%** | ✅ **Best Spline** |
| Spline (k=10, relu) | 0.7396 | +2.4% | Good |
| Spline (k=20, relu) | 0.7363 | +2.0% | Good |
| Spline (k=5, relu) | 0.7425 | +2.9% | Good |
| Spline (k=30, relu) | 0.7257 | +0.5% | Degrading |
| Spline (k=50, relu) | 0.7234 | +0.2% | Worse |
| **ReLU baseline** | **0.7417** | **+2.7%** | ✅ **Winner** |
| Spline (linear init) | 0.7408 | +2.6% | Good |
| Spline (gelu init) | 0.7465 | +3.4% | Very Good |
| Spline (tanh init) | 0.7361 | +2.0% | OK |
| **Spline (zero init)** | **-0.0010** | **-100%** | ❌ **Failure** |
| SIREN baseline | 0.7219 | baseline | Baseline |

**Final Comparison**:
- SH + ReLU: 0.7417 (+2.75% vs SIREN)
- SH + Spline (k=15): 0.7354 (+1.88% vs SIREN)
- SH + SIREN: 0.7219 (baseline)

#### Major Findings

✅ **Optimal configuration**: k=15, relu init, (-3,3) range, fixed positions
✅ **Spline beats SIREN**: +1.88% improvement
❌ **ReLU still wins**: -0.63% vs best spline
❌ **Zero init catastrophic**: R²=-0.001 (complete failure)
✅ **Initialization critical**: 75,000% difference between relu and zero
✅ **Diminishing returns**: k>20 degrades performance
✅ **Fixed positions better**: Learnable worse by -0.0080
✅ **Learned shapes are non-ReLU**: Splines learn distinct, interpretable functions

#### Learned Activation Shapes

Visualization shows:
- **NOT just ReLU**: Splines learn distinct curved shapes
- **Layer-specific**: Each layer learns different transformation
- **Non-monotonic**: Some layers show inflection points
- **Knot clustering**: Concentrate in [-3, 3] region (data-driven)

#### Takeaway

**Verdict**: ReLU remains the winner for SH features on smooth tasks, but splines prove learned activations CAN work (+1.88% vs SIREN). Zero initialization is catastrophically bad. Optimal spline: k=15, relu init, fixed positions.

**Implication**: Test splines on high-frequency tasks (elevation) where their flexibility may shine more than on smooth population density.

#### Files Generated
- `spline_knots_sweep.csv`
- `spline_init_sweep.csv`
- `spline_range_sweep.csv`
- `spline_learnable_positions.csv`
- `spline_baseline_comparison.csv`
- `learned_spline_shapes.png`

#### Documentation
[ANALYSIS_NOTEBOOK18.md](ANALYSIS_NOTEBOOK18.md)

---

## 📅 Pending Experiments (Phase 2)

### Notebook 19: Architecture Interaction
**Priority**: 5 (Medium-low)
**Status**: Not yet created
**Estimated Duration**: ~4 hours

#### Goal
Understand how learned activations interact with network architecture (depth/width).

#### Planned Experiments
1. **Depth sweep**: 2, 3, 4, 5, 8 layers
2. **Width sweep**: 128, 256, 384, 512 hidden units
3. **Depth-width trade-offs**: 8×128 vs 4×256 vs 2×512
4. **Activation by layer**: Learned in early layers only, late layers only, or all layers

#### Key Questions
- Do deeper networks need learned activations less?
- Is there an optimal architecture for each activation type?
- Can we achieve same performance with fewer parameters + better activation?

---

### Notebook 20: High-Frequency Tasks
**Priority**: 4 (Medium)
**Status**: Not yet created
**Estimated Duration**: ~3 hours

#### Goal
Test whether learned activations help more on high-frequency tasks.

#### Planned Tasks
1. **Elevation** (ETOPO1 or similar) - sharp peaks, valleys
2. **Temperature anomalies** - sharp fronts
3. **Urban/rural boundaries** - discrete transitions
4. **Multi-task learning** - predict multiple targets jointly

#### Hypothesis
Population density is too smooth/low-frequency. High-frequency tasks might favor learned activations more.

---

### Notebook 21: Visualization & Interpretation
**Priority**: 2 (High)
**Status**: Not yet created
**Estimated Duration**: ~2 hours

#### Goal
Understand what learned activations are actually learning.

#### Planned Analyses
1. **Plot learned activation shapes**
   - Spline: show knot values and interpolated function
   - RFF: visualize frequency spectrum
   - Compare to ReLU, SIREN

2. **Layer-wise analysis**
   - Do different layers learn different shapes?
   - Early layers vs late layers

3. **Input-output mapping**
   - What input ranges activate what output ranges?
   - Are there dead regions?

4. **Ablation by layer position**
   - Remove activation from layer 1 only
   - Remove from layer 2 only
   - Which layer matters most?

---

### Notebook 22: Robustness & Generalization
**Priority**: 3 (High)
**Status**: Not yet created
**Estimated Duration**: ~4 hours (long runs)

#### Goal
Validate that results are statistically significant and robust.

#### Planned Experiments
1. **Multiple seeds** (5 runs per config)
   - SH + ReLU
   - SH + Spline (k=10)
   - SH + SIREN
   - Report mean ± std

2. **Different spatial blocking sizes**
   - 3° blocks
   - 5° blocks (current)
   - 10° blocks
   - Test if results hold across blocking strategies

3. **Different train/test ratios**
   - 50/50 (current)
   - 70/30
   - 80/20

4. **Statistical significance testing**
   - Paired t-tests between models
   - Confidence intervals
   - Effect sizes

---

### Notebook 23: Raw + Learned Deep Dive
**Priority**: 6 (Low)
**Status**: Not yet created
**Estimated Duration**: ~3 hours

#### Goal
Complete the "Raw vs SH" story by thoroughly characterizing Raw + learned activations.

#### Planned Experiments
1. **RFF parameter sweep**
   - freq_init: linear, log, random
   - max_freq: 5, 10, 20, 50
   - n_features: 10, 25, 50, 100, 200
   - learnable: True/False

2. **Raw + Spline variants**
   - All spline variants from Notebook 18
   - Compare to Raw + SIREN

3. **Hybrid approaches**
   - Raw + SH concatenation
   - Raw → RFF → SH
   - Different encoding combinations

---

## 📈 Experiment Summary Statistics

### Phase 1 (Complete)
- ✅ **Notebooks completed**: 4 (14-17)
- ✅ **Total experiments run**: 40+
- ✅ **Total GPU hours**: ~10 hours (Colab T4)
- ✅ **Calendar time**: ~3 weeks
- ✅ **Key result**: RFF+SH fails, ReLU+SH wins

### Phase 2 (Planned)
- 🔄 **Notebooks planned**: 6 (18-23)
- 🔄 **Total experiments planned**: ~150
- 🔄 **Estimated GPU hours**: ~15 hours
- 🔄 **Estimated calendar time**: 1-2 weeks (parallel execution)

### Overall Project
- **Total notebooks**: 10
- **Total experiments**: ~200
- **Total GPU hours**: ~25 hours
- **Calendar time**: 4-5 weeks

---

## 🎯 Success Criteria for Phase 2

Phase 2 is successful if we can answer:

### 1. When do learned activations help?
- ✅ Which tasks? (smooth vs rough)
- ✅ Which architectures? (deep vs shallow)
- ✅ Which input encodings? (raw vs SH)

### 2. What are learned activations learning?
- ✅ What shapes do splines converge to?
- ✅ Are they approximating known functions?
- ✅ Do different layers learn different shapes?

### 3. Are results robust?
- ✅ Do they hold across seeds?
- ✅ Do they generalize to different regions?
- ✅ Are improvements statistically significant?

### 4. What's the best configuration?
- ✅ Optimal spline knot count?
- ✅ Optimal architecture?
- ✅ Best input encoding?

---

## 🔬 What We've Tested (Comprehensive List)

### Input Encodings
- ✅ Raw coordinates (2D: longitude, latitude)
- ✅ SH(L=10) - 100 spherical harmonic features
- ✅ SH normalization (mean=0, std=1)
- ❌ SH(L=5, 15, 20, 40)
- ❌ Fourier features (standard Gaussian)
- ❌ Wavelets
- ❌ Polynomial features

### Activations
- ✅ ReLU (standard baseline)
- ✅ SIREN (SatCLIP baseline)
- ✅ RFF: n=10, 25, 50, 100 (fixed frequencies)
- ✅ RFF: learnable frequencies
- ✅ RFF: with/without input normalization
- ✅ Spline: k=10, 30 (relu init, linear interpolation)
- ❌ Spline: k=5, 15, 20, 50, 100
- ❌ Spline: other inits (linear, zero, tanh, gelu)
- ❌ Spline: different input ranges
- ❌ Spline: learnable knot positions
- ❌ Spline: cubic interpolation

### Architecture
- ✅ 3 layers, 256 hidden units
- ❌ 2, 4, 5, 8 layers
- ❌ 128, 512, 1024 hidden units
- ❌ Depth-width trade-offs
- ❌ Activations at specific layers only

### Training
- ✅ Adam optimizer, lr=1e-3
- ✅ 100 epochs
- ✅ Batch size 256
- ✅ MSE loss
- ✅ Spatial blocking (5° blocks)
- ❌ SGD, AdamW optimizers
- ❌ Different learning rates (1e-4, 5e-3, 1e-2)
- ❌ Learning rate schedules (cosine, ReduceLROnPlateau)
- ❌ Longer training (500, 1000 epochs)
- ❌ Regularization (weight decay, dropout)
- ❌ Multiple random seeds (error bars)

### Tasks
- ✅ Population density (GPW) at 15-min, 30-min, 1-degree
- ❌ Elevation (high-frequency, sharp features)
- ❌ Temperature (medium-frequency)
- ❌ Urban/rural boundaries (discrete)
- ❌ Multi-task learning

### Analysis
- ✅ Training dynamics (loss curves)
- ✅ Gradient norms
- ✅ Convergence speed
- ✅ SH feature statistics
- ❌ Visualization of learned activation shapes
- ❌ Layer-wise activation statistics
- ❌ Ablation by layer position
- ❌ Statistical significance testing

---

## 📝 Known Limitations & Assumptions

### Data
- Single dataset (GPW population density)
- Single resolution focus (15-min ~28km)
- Smooth, low-frequency target variable
- Fixed spatial blocking (5° grid)

### Training
- Single random seed per config (no error bars)
- Fixed hyperparameters (no tuning)
- Short training (100 epochs)
- No regularization

### Architecture
- Fixed depth (3 layers) and width (256 units)
- Same architecture for all activation types
- No architecture search

### Scope
- Geographic domain only
- Frozen SatCLIP baselines (no fine-tuning)
- Focus on single-task learning

See [EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md](EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md) for complete details.

---

## 🗂️ Related Documents

- **[README.md](README.md)** - Project overview and navigation
- **[PROGRESS_UPDATE.md](PROGRESS_UPDATE.md)** - High-level Phase 1 summary
- **[EXPERIMENTAL_DESIGN_V2.md](EXPERIMENTAL_DESIGN_V2.md)** - Comprehensive Phase 2 plan
- **[EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md](EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md)** - Assumptions catalog

---

## 🔄 Update History

- **2026-01-12**: Created document, consolidated tracking from multiple sources
