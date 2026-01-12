# Learned Activations for Geographic Encoders

**Project Goal**: Investigate whether learned activation functions (RFF, Spline) can improve upon SIREN's sinusoidal activations for geographic coordinate encoding.

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🔄

---

## Quick Navigation

### 📊 Current Status & Next Steps
- **[EXPERIMENTS_TRACKER.md](EXPERIMENTS_TRACKER.md)** - Living document: What's done, pending, and planned
- **[PROGRESS_UPDATE.md](PROGRESS_UPDATE.md)** - High-level Phase 1 summary and Phase 2 roadmap

### 📓 Complete Experiment Notebooks

**Phase 0: Foundations**
| Notebook | Status | Purpose | Key Results |
|----------|--------|---------|-------------|
| **09** | ✅ Complete | Architecture sweep (L=10 vs L=40) | L=10: 0.759 (global), L=40: 0.544 (USA regional) |

**Phase 1a: Initial Attempts (10-12)**
| Notebook | Status | Purpose | Key Results |
|----------|--------|---------|-------------|
| **10** | ✅ Complete | Learned acts v1 (broken baseline) | Direct+Learned: 0.797 (invalid comparison) |
| **11** | ✅ Complete | Learned acts v2 (fair comparison) | Direct+Learned beats L=10 by 4-12% |
| **12** | ✅ Complete | Learned acts v3 (spatial blocking) | Confirms learned acts work properly |

**Phase 1b: Core Comparisons (13-17)**
| Notebook | Status | Purpose | Key Results | Documentation |
|----------|--------|---------|-------------|---------------|
| **13** | ✅ Complete | Phase 1 Core 2×2 grid | SH+Spline: +0.56%, **SH+RFF: -7.96%** | - |
| **14** | ✅ Complete | Spline vs RFF MVP (CPU) | RFF: 0.743 > Spline: 0.735 on raw coords | - |
| **15** | ✅ Complete | Multi-resolution comparison | Raw+RFF beats SatCLIP at ALL resolutions | [ANALYSIS_NOTEBOOK15.md](ANALYSIS_NOTEBOOK15.md) |
| **16** | ✅ Complete | Phase 1 SH combinations (12 models) | **SH+ReLU: +0.63%, SH+RFF: -7.96%** | [CRITICAL_ANALYSIS_NB16.md](CRITICAL_ANALYSIS_NB16.md) |
| **17** | ✅ Complete | Diagnostic: Why RFF+SH fails | Normalization worse (-8.67%), frequency interference | [DIAGNOSTIC_CONCLUSIONS_NB17.md](DIAGNOSTIC_CONCLUSIONS_NB17.md) |

**Phase 2: Deep Characterization (18+)**
| Notebook | Status | Purpose | Key Results | Documentation |
|----------|--------|---------|-------------|---------------|
| **18** | ✅ Complete | Spline deep dive | Optimal: k=15, relu init; ReLU wins (0.7417) | [ANALYSIS_NOTEBOOK18.md](ANALYSIS_NOTEBOOK18.md) |

### 📋 Key Documents

#### Phase 1 Analysis (Complete)
- **[CRITICAL_ANALYSIS_NB16.md](CRITICAL_ANALYSIS_NB16.md)** - Comprehensive analysis of core 2×2 experiment
  - Why RFF+SH catastrophically failed (-7.96%)
  - Why ReLU beat SIREN (+0.63%)
  - Spline performance (+0.56%)

- **[DIAGNOSTIC_CONCLUSIONS_NB17.md](DIAGNOSTIC_CONCLUSIONS_NB17.md)** - Investigation of RFF failure
  - Normalization didn't help (made it worse)
  - Learnable frequencies catastrophic (-18.2%)
  - Frequency interference confirmed

- **[ANALYSIS_NOTEBOOK15.md](ANALYSIS_NOTEBOOK15.md)** - Multi-resolution baseline
  - Raw+RFF/Spline beat SatCLIP L=10
  - L=40 mysteriously underperformed

#### Phase 2 Analysis (In Progress)
- **[ANALYSIS_NOTEBOOK18.md](ANALYSIS_NOTEBOOK18.md)** - Spline deep dive results
  - Optimal configuration: k=15, relu init, fixed positions
  - ReLU beats splines (-0.63% difference)
  - Zero initialization catastrophic failure

#### Experimental Design & Planning
- **[EXPERIMENTAL_DESIGN_V2.md](EXPERIMENTAL_DESIGN_V2.md)** - Comprehensive Phase 2 plan
  - 6 new notebooks (18-23)
  - ~150 experiments planned
  - Priority ordering with rationale

- **[EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md](EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md)** - Complete catalog
  - All assumptions made
  - What's been tested (50+ experiments across 10 notebooks)
  - What hasn't been tested
  - Known limitations

- **[CLAUDE_GUIDE.md](CLAUDE_GUIDE.md)** - Standardized workflow for analyzing results
  - Step-by-step process for ingesting new notebooks
  - Templates for analysis documents
  - Ensures consistency across analyses

---

## Executive Summary: Complete Experimental Arc (NB09-18)

### 🔬 Phase 0: Architecture Foundations (NB09)

**Goal**: Does MLP architecture affect L=10 vs L=40 performance?

**Key Baselines** (Population density, 15-min resolution):
- **Global**: L=10 = 0.759, L=40 = 0.683 (L=10 wins +7.6%)
- **USA**: L=10 = 0.476, L=40 = 0.544 (L=40 wins +6.9%)
- **Europe**: L=10 = 0.590, L=40 = 0.650 (L=40 wins +6.0%)
- **China**: L=10 = 0.829, L=40 = 0.855 (L=40 wins +2.5%)

**Finding**: L=40's regional advantage persists across ALL architectures → it's in the embeddings, not downstream MLP

### 🔄 Phase 1a: Iterative Development (NB10-12)

**NB10** (Initial attempt): ❌ Broken - MLP overfits on frozen SatCLIP embeddings
**NB11** (Fixed): ✅ Direct+Learned beats SatCLIP L=10 by +4-12% across regions
**NB12** (Spatial blocking): ✅ Confirms learned acts work with proper train/test splits

### 🎯 Phase 1b: Core 2×2 Grid (NB13-16)

**The Critical Comparison**:

| | SIREN | Learned Acts |
|---|-------|--------------|
| **Raw (lon,lat)** | 0.743 (baseline) | RFF: 0.735, Spline: 0.735 (-1%) |
| **SH(L=10)** | 0.743 (baseline) | **RFF: 0.663 (-10.8%)**, **Spline: 0.748 (+0.7%)** |

**Winner**: SH + ReLU (0.749, +0.8% vs SIREN) - simplest solution

### 🔍 Phase 1c: Diagnostics (NB17-18)

**NB17** - Why does RFF+SH fail?
- ❌ Normalization → worse (-8.67%)
- ❌ Learnable frequencies → catastrophic (-18.2%)
- ✅ **Root cause**: Frequency interference (SH harmonics ≠ Cartesian Fourier)

**NB18** - Spline characterization (complete)
- ✅ Optimal: k=15 knots, relu init, (-3,3) range, fixed positions
- ✅ ReLU still wins (0.7417 vs best spline 0.7354)
- ❌ Zero init catastrophic failure (R²=-0.001)

### 🏆 Final Winners by Configuration

| Input | Activation | R² (best) | vs SIREN | Status |
|-------|------------|-----------|----------|--------|
| **SH(L=10)** | **ReLU** | **0.7490** | **+2.93%** | ✅ **Winner** |
| **SH(L=10)** | **Spline (k=10)** | **0.7483** | **+2.53%** | ✅ **Good** |
| SH(L=10) | SIREN | 0.7427 | baseline | ✅ Baseline |
| Raw (2D) | RFF (n=25) | 0.7426 | -0.01% | ✅ Works |
| Raw (2D) | Spline (k=10) | 0.7350 | -0.77% | ✅ OK |
| **SH(L=10)** | **RFF (n=25)** | **0.6631** | **-7.74%** | ❌ **Failed** |

### 💡 Key Insights

1. **Frequency Interference is Real**
   - SH (spherical harmonics) and RFF (Cartesian Fourier) use incompatible frequency representations
   - They conflict during optimization → catastrophic failure
   - Confirmed through: normalization (-8.67%), learnable frequencies (-18.2%), gradient analysis

2. **Simple Often Wins**
   - ReLU beats both SIREN and learned activations with SH features
   - SH already provides frequencies → just need nonlinearity
   - No hyperparameters to tune, Kaiming init works perfectly

3. **Splines Work Well**
   - Local, piecewise approximation doesn't interfere with SH
   - Small but consistent improvement over SIREN (+2.5%)
   - Simpler optimization than RFF (lower gradient norms)

4. **Task May Matter**
   - Population density is smooth/low-frequency
   - High-frequency tasks (elevation, edges) might favor learned activations more
   - Need to test on different data

### 🚫 What NOT to Do

❌ **NEVER use RFF + SH features** - fundamentally broken, not fixable
❌ Don't over-engineer - if ReLU works, use ReLU
❌ Don't mix frequency-based inputs with frequency-based activations

### ✅ Recommendations

**For SH-encoded inputs:**
- Default: **SH + ReLU** (simple, robust, best performance)
- If you want learned activations: **SH + Spline** (+0.5% improvement)

**For raw coordinates:**
- If you need frequency discovery: **Raw + SIREN** (still best)
- For local adaptivity: **Raw + Spline** (comparable to SIREN)
- **Raw + RFF** works but underperforms SIREN

---

## Phase 2 Roadmap

See **[EXPERIMENTAL_DESIGN_V2.md](EXPERIMENTAL_DESIGN_V2.md)** for full details.

### Recently Completed: Notebook 18 - Spline Deep Dive ✅

**Status**: Complete (2026-01-12)

**Experiments Completed**:
1. Capacity analysis (knot count: 5, 10, 15, 20, 30, 50) ✅
2. Initialization strategies (relu, linear, zero, tanh, gelu) ✅
3. Input range sensitivity (-3,3 vs -5,5 vs -10,10) ✅
4. Learnable knot positions (fixed vs learnable) ✅
5. Interpolation methods (linear vs cubic) ✅
6. Visualization of learned activation shapes ✅

**Key Findings**:
- Optimal: k=15, relu init, (-3,3) range, fixed positions
- ReLU still wins (0.7417 vs 0.7354)
- Zero init catastrophic (R²=-0.001)

**Full Analysis**: [ANALYSIS_NOTEBOOK18.md](ANALYSIS_NOTEBOOK18.md)

### Upcoming Notebooks

| Priority | Notebook | Focus | Goal |
|----------|----------|-------|------|
| 1 | **19** | **Simplicity bias tests** | **Find "alpha" - where learned acts excel** |
| 2 | **20** | Complexity analysis + visualization | What are splines learning? |
| 3 | **21** | Robustness | Multiple seeds, error bars |
| 4 | **22** | Architecture interaction | Depth/width sweeps |
| 5 | **23** | Raw + Learned | Complete the story |

---

## Project Structure

```
hamza/
├── README.md                                    ← You are here
├── EXPERIMENTS_TRACKER.md                       ← Living tracker (what's done/pending)
├── PROGRESS_UPDATE.md                           ← High-level summary
│
├── Notebooks/
│   ├── 09_architecture_sweep.ipynb              ✅ Complete (baselines)
│   ├── 10_learned_activations.ipynb             ✅ Complete (broken)
│   ├── 11_learned_activations_v2.ipynb          ✅ Complete (fixed)
│   ├── 12_learned_activations_v3.ipynb          ✅ Complete (spatial blocking)
│   ├── 13_phase1_core_comparison.ipynb          ✅ Complete (2×2 grid)
│   ├── 14_spline_vs_rff_simple.ipynb            ✅ Complete (MVP)
│   ├── 15_multi_resolution_comparison.ipynb     ✅ Complete
│   ├── 16_phase1_sh_combinations.ipynb          ✅ Complete
│   ├── 17_diagnostic_rff_failure.ipynb          ✅ Complete
│   └── 18_spline_deep_dive.ipynb                ✅ Complete
│
├── Analysis Documents/
│   ├── CRITICAL_ANALYSIS_NB16.md                ← Core 2×2 results
│   ├── DIAGNOSTIC_CONCLUSIONS_NB17.md           ← Why RFF fails
│   ├── ANALYSIS_NOTEBOOK15.md                   ← Multi-resolution baseline
│   └── ANALYSIS_NOTEBOOK18.md                   ← Spline deep dive
│
├── Planning Documents/
│   ├── EXPERIMENTAL_DESIGN_V2.md                ← Phase 2 comprehensive plan
│   ├── EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md ← What we've tested
│   └── CLAUDE_GUIDE.md                          ← Workflow for analyzing results
│
├── Meta-Documentation/
│   └── ORGANIZATION_SUMMARY.md                  ← Documentation organization
│
└── Archive/ (outdated, superseded)
    ├── meeting_notes.md                         ← Initial brainstorming
    ├── plan.md                                  ← Superseded by EXPERIMENTAL_DESIGN_V2.md
    ├── VARIANTS_TO_TEST.md                      ← Incorporated into EXPERIMENTAL_DESIGN_V2.md
    ├── ARCHITECTURE_SETUP.md                    ← Setup notes
    └── EXPERIMENT_ROADMAP.md                    ← Superseded by EXPERIMENTAL_DESIGN_V2.md
```

**Note**: The [Archive](Archive/) folder contains documents that have been superseded by newer, more comprehensive documents. They're kept for historical reference.

---

## Separate Project: SatCLIP Resolution Investigation

**Location**: [satclip_research.md](satclip_research.md)

**Notebooks**: 00-07 (plus notebook 09 provides additional MLP architecture context)

**Purpose**: Investigate the effective spatial resolution of SatCLIP's location encoder (L=10 vs L=40)

**Status**: 7 notebooks complete (00-07), plus architecture sweep (09)

**Key Finding**: L=40 has 2× better effective resolution (~100km vs ~200km) but fails at regression tasks globally while excelling regionally

**This is a SEPARATE project** from the learned activations work and has its own comprehensive documentation.

---

## Getting Started

1. **New to the project?** Start with [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md) for a high-level overview
2. **Want to understand Phase 1 results?** Read [CRITICAL_ANALYSIS_NB16.md](CRITICAL_ANALYSIS_NB16.md)
3. **Planning next experiments?** Check [EXPERIMENTS_TRACKER.md](EXPERIMENTS_TRACKER.md) and [EXPERIMENTAL_DESIGN_V2.md](EXPERIMENTAL_DESIGN_V2.md)
4. **Looking for specific assumptions?** See [EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md](EXPERIMENTAL_ASSUMPTIONS_AND_COVERAGE.md)
5. **Want the complete experimental timeline?** See [EXPERIMENTS_TRACKER.md](EXPERIMENTS_TRACKER.md) starting from Notebook 09

---

## Key Metrics

### Phase 0+1 Progress (Complete)
- ✅ 9 notebooks completed (09-17)
- ✅ 50+ experiments run
- ✅ ~15 hours GPU time (Colab T4)
- ✅ Core hypotheses tested:
  - ✅ L=40 regional advantage is fundamental (not architecture-dependent)
  - ✅ Learned acts can discover frequencies from raw coords
  - ✅ SH+ReLU wins overall (+2.93% vs SIREN)
  - ✅ RFF+SH catastrophically fails (-7.74% to -10.8%)
  - ✅ Frequency interference is the root cause

### Phase 2 Progress (In Progress)
- ✅ 1/6 notebooks completed (18)
- ✅ 20 experiments run (6 experiment groups)
- ✅ ~0.5 hours GPU time (Colab T4)
- ✅ Spline characterization complete
- 🔄 5 notebooks remaining (19-23)

---

## Contributors

- **Hamza** - Experimental execution and analysis
- **Dan** - Project guidance and hypothesis formulation

---

## Last Updated

2026-01-12 - Notebook 18 complete: Spline deep dive analysis integrated into documentation
