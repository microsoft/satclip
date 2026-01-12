# Learned Activations Experiment Roadmap

## Core Question

**Can learned activation functions match or beat SH + SIREN with better parameter efficiency?**

Win condition: **Performance per parameter** (not training time)

---

## The 2x2 Core Comparison

| | SIREN | Learned Acts |
|---|-------|--------------|
| **Raw coords** | Baseline: SIREN discovers frequencies | Test: Can learned acts do it too? |
| **SH features** | Baseline: SH provides frequencies | Test: Better nonlinearity than SIREN? |

This 2x2 grid is the heart of the experiment.

---

## Phased Approach

### Phase 1: Core 2x2 (START HERE)
**Goal**: Establish whether learned activations work at all

**Setup**:
- Task: Population density regression (supervised, single modality)
- Data: GPW 15-min resolution
- Layers: 3 (simple)
- Encoding: Direct coords OR SH(L=10)
- Spatial blocking: Yes (5° grid)

**Models to compare**:
1. `Raw + SIREN` - Direct coords with SIREN (w0_init=30, w0=1)
2. `Raw + Learned` - Direct coords with Fourier-parameterized activations
3. `SH + SIREN` - SH(L=10) features with SIREN
4. `SH + Learned` - SH(L=10) features with learned activations

**Metrics**:
- R² score
- Parameter count
- R² per 10K parameters (efficiency)

**Expected outcome**:
- If `Raw + Learned` ≈ `Raw + SIREN`: Learned acts can discover frequencies
- If `SH + Learned` > `SH + SIREN`: Better nonlinearity

---

### Phase 2: Activation Type Ablation
**Goal**: Find best activation parameterization

**Activation types to test**:
1. **Fourier/RFF** (current): `g(x) = Σ a_k sin(ω_k x) + b_k cos(ω_k x)`
2. **B-splines** (new): Piecewise polynomial, locally supported
3. **Polynomial** (simple baseline): `g(x) = Σ c_k x^k`

**Why splines might be better**:
- Local support = sparse gradients
- Easier to jointly optimize
- Dan mentioned "library of RFFs is cleaner than splines, but needs verification"

**Test on**: Best config from Phase 1

---

### Phase 3: Architecture Ablation
**Goal**: Find optimal depth/width

**Variables**:
- Layers: 3 vs 5 vs 8
- Frequencies: 10 vs 25 vs 50
- Hidden dim: 128 vs 256 vs 512

**Keep fixed**: Best activation type from Phase 2

---

### Phase 4: Spatial Variation
**Goal**: Test if location-adaptive activations help

**Compare**:
- Global learned activations (same everywhere)
- Spatially-varying (MoE with shared gate)

**Test specifically on regional tasks** where we expect spatial variation to matter.

---

### Phase 5: Scale Up
**Goal**: Full contrastive training

**Steps**:
1. Test on MOSAIKS data (if available)
2. Multi-task: Population + Temperature + Elevation
3. Contrastive training with satellite imagery
4. Compare to full SatCLIP

---

## Phase 1 Implementation Details

### Minimal Config

```python
# Core settings
N_LAYERS = 3
HIDDEN_DIM = 256
OUTPUT_DIM = 256
N_FREQUENCIES = 25  # For learned activations

# SIREN settings (match SatCLIP)
W0_INITIAL = 30.0   # First layer
W0 = 1.0            # Subsequent layers

# Training
N_SAMPLES = 15000
EPOCHS = 100
BATCH_SIZE = 256
GRID_SIZE = 5.0     # Spatial blocking
```

### Models

```python
# 1. Raw + SIREN
class RawSIREN:
    input: (lon, lat) normalized to [-1, 1]
    layers: SirenLayer(2, 256, w0=30) -> SirenLayer(256, 256, w0=1) x2 -> Linear(256, 256)

# 2. Raw + Learned
class RawLearned:
    input: (lon, lat) normalized to [-1, 1]
    layers: Linear -> LearnedAct -> Linear -> LearnedAct -> Linear -> LearnedAct -> Linear

# 3. SH + SIREN
class SHSiren:
    input: SH(lon, lat) -> 100 features (L=10)
    layers: Same as SatCLIP's SIREN network

# 4. SH + Learned
class SHLearned:
    input: SH(lon, lat) -> 100 features (L=10)
    layers: Linear -> LearnedAct -> Linear -> LearnedAct -> Linear -> LearnedAct -> Linear
```

### Evaluation

```python
# For each model:
results = {
    'model': name,
    'r2': r2_score,
    'params': param_count,
    'efficiency': r2 / (params / 10000),  # R² per 10K params
    'train_time': seconds,
}
```

---

## Decision Tree

```
Phase 1 Results
    │
    ├── Raw+Learned ≈ Raw+SIREN?
    │       │
    │       ├── YES → Learned acts can discover frequencies!
    │       │         → Proceed to Phase 2 (activation types)
    │       │
    │       └── NO → Learned acts need SH features
    │                → Focus on SH+Learned path
    │
    └── SH+Learned > SH+SIREN?
            │
            ├── YES → Better nonlinearity confirmed!
            │         → Scale up in Phase 5
            │
            └── NO → SIREN might be optimal for SH
                     → Try different activation types (Phase 2)
```

---

## Files to Create

1. `13_phase1_core_comparison.ipynb` - Clean 2x2 comparison
2. `14_phase2_activation_types.ipynb` - Splines vs Fourier vs Polynomial
3. `15_phase3_architecture.ipynb` - Depth/width ablation
4. `16_phase4_spatial.ipynb` - Spatially-varying activations
5. `17_phase5_contrastive.ipynb` - Full scale training

---

## Quick Start

**Run Phase 1 first** - this tells us if the whole approach is viable.

If Phase 1 shows promise (learned acts work), then:
- Phase 2 optimizes the activation parameterization
- Phase 3 optimizes the architecture
- Phase 4 adds spatial adaptation
- Phase 5 scales to contrastive

If Phase 1 fails (learned acts don't work), then:
- Investigate why
- Maybe need more frequencies
- Maybe need different init
- Maybe the hypothesis is wrong

---

## Notes from Dan

- Win is performance/parameter, not training time
- Loops learning activations = slower training (expected)
- Can be jointly optimized
- Start supervised (regression), then contrastive
- Library of RFFs might be cleaner than splines (needs verification)
- MOSAIKS data available for single-modality testing
