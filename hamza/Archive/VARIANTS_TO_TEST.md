# Comprehensive Variants Matrix

This document outlines all RFF, Spline, and SH combinations to test based on notebook 15 results and Dan's roadmap.

---

## Current Status (From Notebooks 14-15)

### Notebook 14 Results (15-min, CPU):
- RFF (n=25): R² = 0.743
- Spline (k=10): R² = 0.735
- ReLU: R² = 0.729

### Notebook 15 Results (15-min, GPU):
- RFF (n=25): R² = 0.733
- Spline (k=10): R² = 0.728
- ReLU: R² = 0.722
- **SatCLIP L=10: R² = 0.704** (-2.9% vs RFF)
- **SatCLIP L=40: R² = 0.618** (-11.5% vs RFF!)

**Key finding**: Our simple learned activations beat SatCLIP across all resolutions!

---

## Phase 1: RFF Variants (Notebook 16)

### Core RFF Parameters:
```python
class RFFActivation(nn.Module):
    def __init__(self,
                 n_features=25,      # Number of Fourier features
                 max_freq=10.0,      # Maximum frequency
                 learnable_freq=False, # Learn frequencies?
                 freq_init='linear'): # Frequency initialization
```

### Variants to Test:

#### 1. **Number of Features** (most important)
- n=10: Minimal
- n=25: Current baseline
- n=50: Dan suggested
- n=100: Dan suggested
- n=200: High capacity

**Hypothesis**: More features → more expressiveness, but risk overfitting

#### 2. **Learnable vs Fixed Frequencies**
- Fixed (current): `self.register_buffer('freqs', freqs)`
- Learnable: `self.freqs = nn.Parameter(freqs)`

**Hypothesis**: Learning ω might help adapt to task-specific frequency scales

#### 3. **Frequency Initialization**
- **Linear** (current): `ω ∈ [0.1, 10]` evenly spaced
- **Log-spaced**: `ω ∈ [10^-2, 10^2]` logarithmic spacing
- **Random**: `ω ~ U(0, max_freq)` uniform random

**Hypothesis**: Log-spaced covers more frequency scales, might generalize better

#### 4. **Max Frequency Range**
- max_freq=5: Conservative
- max_freq=10: Current
- max_freq=20: Higher frequency
- max_freq=50: Very high frequency

**Hypothesis**: Task-dependent optimal range

### Full RFF Matrix (60 combinations):
| n_features | learnable_freq | freq_init | max_freq |
|-----------|----------------|-----------|----------|
| 25 | False | linear | 10 | ← **Current baseline**
| 50 | False | linear | 10 | ← **Notebook 16**
| 100 | False | linear | 10 | ← **Notebook 16**
| 25 | True | linear | 10 | ← Test if learning helps
| 50 | True | linear | 10 |
| 25 | False | log | 10 | ← Test initialization
| 50 | False | log | 10 |
| 25 | False | linear | 20 | ← Test frequency range
| 50 | False | linear | 20 |

**Priority order**: n_features > freq_init > max_freq > learnable_freq

---

## Phase 2: Spline Variants (Notebook 17)

### Core Spline Parameters:
```python
class SplineActivation(nn.Module):
    def __init__(self,
                 n_knots=10,             # Number of control points
                 input_range=(-3, 3),    # Input range
                 init='relu'):           # Initialization
```

### Variants to Test:

#### 1. **Number of Knots** (most important)
- k=5: Minimal
- k=10: Current baseline
- k=20: Medium
- k=30: High (notebook 16)
- k=50: Very high

**Hypothesis**: More knots → more expressive, but more parameters

#### 2. **Initialization Strategy**
- **ReLU** (current): `knot_y = relu(knot_x)`
- **Linear**: `knot_y = knot_x` (identity)
- **Zero**: `knot_y = 0` (learn from scratch)
- **Random**: `knot_y ~ N(0, 0.1)`

**Hypothesis**: ReLU init helps because target functions are often positive

#### 3. **Input Range**
- (-3, 3): Current (covers ~99.7% of normalized data)
- (-5, 5): Wider
- (-10, 10): Very wide

**Hypothesis**: Wider range prevents clamping at extremes

#### 4. **Spline Type** (advanced)
Currently using **linear interpolation** between knots. Could try:
- **Cubic spline**: Smooth derivatives
- **Hermite spline**: Specify derivatives at knots
- **B-spline**: Better numerical properties

#### 5. **Learnable Knot Positions**
Currently knot positions are fixed, only values are learned. Could learn both:
```python
self.knot_x = nn.Parameter(knot_x)  # Learn positions too
self.knot_y = nn.Parameter(knot_y)
```

**Hypothesis**: Adaptive knot spacing might help

### Full Spline Matrix (45 combinations):
| n_knots | init | input_range | learnable_pos |
|---------|------|-------------|---------------|
| 10 | relu | (-3, 3) | False | ← **Current baseline**
| 20 | relu | (-3, 3) | False | ← **Notebook 16**
| 30 | relu | (-3, 3) | False | ← **Notebook 16**
| 10 | linear | (-3, 3) | False | ← Test initialization
| 20 | linear | (-3, 3) | False |
| 10 | relu | (-5, 5) | False | ← Test range
| 20 | relu | (-5, 5) | False |
| 10 | relu | (-3, 3) | True | ← Test learnable positions
| 20 | relu | (-3, 3) | True |

**Priority order**: n_knots > init > input_range > learnable_pos

---

## Phase 3: SH Combinations (Notebook 16)

### Spherical Harmonics Input:

SH encodes (lon, lat) into high-dimensional features capturing spatial frequencies.

**Output dimension**: `(L+1)² features`
- L=10 → 121 features (but SatCLIP uses 100?)
- L=40 → 1681 features (but SatCLIP uses 1600?)

### The 2×2 Grid (Dan's Core Question):

| Input | Activation | Description |
|-------|------------|-------------|
| Raw (2D) | SIREN | Baseline: SIREN discovers frequencies |
| Raw (2D) | RFF/Spline | **Q1**: Can learned acts discover frequencies? |
| SH(L=10) | SIREN | SatCLIP baseline |
| SH(L=10) | RFF/Spline | **Q2**: Better nonlinearity than SIREN? |

### Extended Grid (Notebook 16):

#### Encoding Options:
1. **Raw coordinates**: (lon, lat) ∈ ℝ²
2. **SH(L=10)**: 100 features
3. **SH(L=40)**: 1600 features (optional)

#### Activation Options:
1. **SIREN**: SatCLIP baseline
2. **ReLU**: Simple baseline
3. **RFF**: n=25, 50, 100
4. **Spline**: k=10, 20, 30

### Full Combination Matrix (Notebook 16):

| Encoding | Activation | Tested in NB16? |
|----------|------------|-----------------|
| Raw | SIREN | ✓ |
| Raw | ReLU | (could add) |
| Raw | RFF (n=25) | ✓ |
| Raw | RFF (n=50) | ✓ |
| Raw | RFF (n=100) | ✓ |
| Raw | Spline (k=10) | ✓ |
| Raw | Spline (k=30) | ✓ |
| SH(L=10) | SIREN | ✓ |
| SH(L=10) | ReLU | ✓ |
| SH(L=10) | RFF (n=25) | ✓ |
| SH(L=10) | RFF (n=50) | ✓ |
| SH(L=10) | Spline (k=10) | ✓ |
| SH(L=10) | Spline (k=30) | ✓ |

Total: **12 models** in notebook 16

---

## Phase 4: Architecture Ablations (Notebook 17)

### Current Architecture:
```python
dims = [input_dim] + [256] * 3 + [256]
# → [input, 256, 256, 256, 256]
```

### Ablations:

#### 1. **Depth** (number of hidden layers)
- 2 layers: Shallow
- 3 layers: Current
- 5 layers: Medium
- 8 layers: Deep

#### 2. **Width** (hidden dimension)
- 128: Narrow
- 256: Current
- 512: Wide
- 1024: Very wide

#### 3. **Output Dimension**
- Current: 256 (matches SatCLIP)
- Could try: 128, 512

### Architecture Matrix:
| Depth | Width | Output | Params (approx) |
|-------|-------|--------|-----------------|
| 3 | 256 | 256 | ~230K | ← Current
| 2 | 256 | 256 | ~200K |
| 5 | 256 | 256 | ~400K |
| 3 | 128 | 256 | ~100K |
| 3 | 512 | 256 | ~800K |
| 5 | 512 | 256 | ~1.5M | ← Match SatCLIP L=40 params

**Goal**: Find optimal depth/width trade-off

---

## Phase 5: Training Hyperparameters (Notebook 17)

### Current Setup:
```python
epochs = 100
batch_size = 256
lr = 1e-3
optimizer = Adam
```

### Ablations:

#### 1. **Learning Rate**
- 1e-4: Conservative
- 1e-3: Current
- 1e-2: Aggressive

#### 2. **Epochs**
- 50: Quick
- 100: Current
- 200: Thorough
- 500: Full convergence

#### 3. **Batch Size**
- 128: Small
- 256: Current
- 512: Large

#### 4. **Learning Rate Schedule**
- Constant (current)
- Cosine annealing
- ReduceLROnPlateau
- Step decay

#### 5. **Optimizer**
- Adam (current)
- AdamW (with weight decay)
- SGD with momentum

---

## Experimental Design Recommendations

### Notebook 16 (Phase 1: Core 2×2 + Variants):
**Goal**: Answer Dan's core question

**Models** (12 total):
1. Raw + SIREN
2-4. Raw + RFF (n=25, 50, 100)
5-6. Raw + Spline (k=10, 30)
7. SH(L=10) + SIREN
8-9. SH(L=10) + RFF (n=25, 50)
10-11. SH(L=10) + Spline (k=10, 30)
12. SH(L=10) + ReLU

**Win conditions**:
- Raw + RFF ≈ Raw + SIREN → Learned acts discover frequencies
- SH + RFF > SH + SIREN → Better nonlinearity

### Notebook 17 (RFF Ablation):
**Goal**: Find optimal RFF hyperparameters

**Variables**:
- n_features: 10, 25, 50, 100, 200
- learnable_freq: True, False
- freq_init: 'linear', 'log', 'random'
- max_freq: 5, 10, 20, 50

**Start with**: n_features sweep (most important)

### Notebook 18 (Spline Ablation):
**Goal**: Find optimal Spline hyperparameters

**Variables**:
- n_knots: 5, 10, 20, 30, 50
- init: 'relu', 'linear', 'zero', 'random'
- input_range: (-3,3), (-5,5), (-10,10)
- learnable_pos: True, False

**Start with**: n_knots sweep (most important)

### Notebook 19 (Architecture Ablation):
**Goal**: Find optimal depth/width

**Variables**:
- depth: 2, 3, 5, 8
- width: 128, 256, 512
- Use best activation from notebooks 16-18

### Notebook 20 (Training Ablation):
**Goal**: Optimize training hyperparameters

**Variables**:
- lr: 1e-4, 1e-3, 1e-2
- epochs: 50, 100, 200, 500
- batch_size: 128, 256, 512
- scheduler: None, cosine, plateau

---

## Summary: Priority Order

### High Priority (Do First):
1. ✅ **Notebook 15**: Multi-resolution comparison (DONE)
2. **Notebook 16**: Phase 1 core 2×2 + variants (READY TO RUN)
3. **Investigate SatCLIP L=40**: Why so bad? Try different Ridge alphas

### Medium Priority (Next):
4. **RFF n_features sweep**: 10, 25, 50, 100, 200
5. **Spline n_knots sweep**: 5, 10, 20, 30, 50
6. **SH(L=40) combinations**: If Phase 1 shows promise

### Lower Priority (Later):
7. **Frequency initialization**: linear vs log vs random
8. **Learnable frequencies**: Does it help?
9. **Architecture ablation**: depth/width
10. **Training ablation**: LR, epochs, schedulers

### Future Work (Phase 4-5):
11. **Spatial gating (MoE)**: Location-adaptive activations
12. **MOSAIKS data**: Test on different task
13. **Contrastive training**: Scale to dual-modality

---

## Open Questions

### From Notebook 15 Results:

1. **Why is SatCLIP L=40 so bad?**
   - Ridge regression can't handle 1600 dims?
   - Features are redundant?
   - Need different regularization?
   - **Action**: Test Ridge with α ∈ {0.01, 0.1, 1.0, 10, 100}

2. **Why did RFF/Spline beat SatCLIP?**
   - Better nonlinearity?
   - Joint optimization vs frozen embeddings?
   - Parameter efficiency?
   - **Action**: Compare learning curves

3. **Is 100 epochs enough?**
   - Learned acts might need more iterations
   - **Action**: Train to 200-500 epochs, check convergence

4. **Are results robust?**
   - Only 1 run per config
   - **Action**: Multiple runs with different seeds

5. **What do learned activations look like?**
   - Visualize activation shapes after training
   - Compare to ReLU and SIREN
   - **Action**: Add visualization to notebook 16

---

## Success Criteria (From Roadmap)

### Phase 1 Win Conditions:
- **If Raw + Learned ≈ Raw + SIREN**: Learned acts can discover frequencies! → Proceed to Phase 2
- **If SH + Learned > SH + SIREN**: Better nonlinearity confirmed! → Scale up in Phase 5

### Overall Win Condition (From Dan):
**Performance per parameter** ≥ SH + SIREN

**Current status**:
- Our Raw + RFF (231K params) beats SatCLIP L=10 (446K params) by 2.9%
- Efficiency: 2× better than L=10, 6× better than L=40
- **Already winning on efficiency!**

Next step: Can we improve absolute R² by combining with SH?
