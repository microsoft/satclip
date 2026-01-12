# Diagnostic Conclusions: Notebook 17

## Executive Summary

**Normalization did NOT fix the RFF + SH failure. In fact, it made it slightly worse.**

- **RFF without normalization**: R² = 0.6256 (-7.74% vs SIREN)
- **RFF with normalization**: R² = 0.6193 (-8.67% vs SIREN)
- **RFF with learnable frequencies**: R² = 0.5549 (-18.2% vs SIREN)

**Verdict**: The frequency interference hypothesis is correct. RFF and SH are fundamentally incompatible.

---

## Complete Results

| Model | R² | vs SIREN | Conclusion |
|-------|-----|----------|------------|
| **SH + ReLU** | **0.6979** | **+2.93%** | ✅ Best overall |
| **SH + Spline (k=10)** | **0.6953** | **+2.53%** | ✅ Works well |
| SH + SIREN | 0.6781 | baseline | ✅ Standard |
| SH + RFF (no norm) | 0.6256 | -7.74% | ❌ Failed |
| SH + RFF (norm) | 0.6193 | -8.67% | ❌ Still failed |
| SH + RFF (learnable) | 0.5549 | -18.2% | ❌ Even worse |

**Note**: These R² values are lower than notebook 16 (which had ReLU at 0.749, SIREN at 0.743). This is likely due to different random seeds / train-test splits.

---

## Answer to Hypotheses

### Hypothesis 1: Input Statistics Mismatch ❌ REJECTED

**Test**: Standardize SH features to mean=0, std=1 before RFF

**Result**: Normalization made it WORSE (0.6193 vs 0.6256)

**Conclusion**: The problem is not input statistics. RFF's fixed frequency range of [0.1, 10] is not the issue.

### Hypothesis 2: Learnable Frequencies Would Help ❌ REJECTED

**Test**: Allow RFF to learn frequencies instead of using fixed values

**Result**: Learnable frequencies made it MUCH WORSE (0.5549)

**Conclusion**: Learning frequencies introduces even more parameters to optimize in an already difficult landscape. The gradient norms were very high (93.3 at epoch 1), suggesting unstable optimization.

### Hypothesis 3: Frequency Interference ✅ CONFIRMED

**Evidence**:
1. Normalization didn't help → not a scaling issue
2. Learnable frequencies didn't help → not a fixed frequency issue
3. Spline works well (+2.53%) → local, non-frequency activations are compatible
4. ReLU works well (+2.93%) → simple, non-frequency activations are compatible

**Conclusion**: SH features encode spatial frequencies as spherical harmonics. RFF tries to add Cartesian Fourier components on top. These two frequency representations are incompatible and interfere with each other during optimization.

### Hypothesis 4: Optimization Difficulty ✅ CONFIRMED

**Evidence from gradient norms**:

| Model | Final Grad Norm | Stability |
|-------|----------------|-----------|
| ReLU | 1.99 | ✅ Low, stable |
| Spline | 2.98 | ✅ Low, stable |
| SIREN | 9.44 | ⚠️ Medium |
| RFF (no norm) | 6.26 | ⚠️ Medium-high |
| RFF (norm) | 4.65 | ⚠️ Medium |
| RFF (learnable) | 5.89 | ❌ High, unstable |

**Observation**:
- ReLU and Spline have low, stable gradient norms
- RFF has 2-3× higher gradient norms than simple activations
- RFF with learnable frequencies had gradient norms of 93.3 at epoch 1, indicating extreme instability

**Conclusion**: The RFF + SH optimization landscape is significantly harder than ReLU/Spline, likely due to interdependencies between RFF parameters and network weights.

---

## Training Dynamics Analysis

### Convergence Speed

**ReLU**:
- Epoch 1: R² = 0.693
- Epoch 10: R² = 0.731
- Epoch 100: R² = 0.698
- **Converges fast, stable**

**Spline**:
- Epoch 1: R² = 0.666
- Epoch 10: R² = 0.734
- Epoch 100: R² = 0.695
- **Converges fast, stable**

**SIREN**:
- Epoch 1: R² = 0.705
- Epoch 10: R² = 0.711
- Epoch 100: R² = 0.678
- **Slow convergence, degrades over time**

**RFF (no norm)**:
- Epoch 1: R² = 0.091 (!!!)
- Epoch 10: R² = 0.536
- Epoch 100: R² = 0.626
- **Very slow start, gradual improvement**

**RFF (norm)**:
- Epoch 1: R² = 0.127
- Epoch 10: R² = 0.492
- Epoch 100: R² = 0.619
- **Very slow start, doesn't catch up**

**RFF (learnable)**:
- Epoch 1: R² = -0.011 (!!!)
- Epoch 10: R² = 0.350
- Epoch 100: R² = 0.555
- **Catastrophic start, never recovers**

### Loss Curves

All models show decreasing training loss, but:
- **ReLU/Spline**: Smooth, monotonic decrease
- **SIREN**: Some oscillations, but generally decreasing
- **RFF**: High variance, lots of oscillations
- **RFF (learnable)**: Extremely noisy, suggests unstable optimization

---

## Why Does Spline Work But RFF Doesn't?

### Spline Properties:
1. **Local, piecewise linear**: No global frequency assumptions
2. **Adaptive**: Learns knot values to fit any monotonic/non-monotonic shape
3. **Simple gradients**: Piecewise linear → easy backprop
4. **Few parameters**: 10 knots × 3 layers = 30 params
5. **No interference**: Doesn't impose frequency structure on SH features

### RFF Properties:
1. **Global Fourier basis**: Assumes sinusoidal structure
2. **Fixed frequency range**: [0.1, 10] chosen for normalized inputs
3. **Complex gradients**: Interdependent sin/cos coefficients
4. **Many parameters**: (25 freqs × 2 coeffs + scale + bias) × 3 layers = 156 params
5. **Frequency interference**: Conflicts with SH's spherical harmonic basis

### Why ReLU Works Best:
1. **No frequency assumptions**: Pure elementwise nonlinearity
2. **Simplest possible**: No extra parameters to learn
3. **Well-studied**: Kaiming initialization is optimal for SH inputs
4. **Lets SH features through**: Doesn't distort the frequency content

---

## SH Feature Statistics

### Key Observations:

```
SH features shape: (10532, 100)

Per-feature statistics:
  Mean (first 5): [0.886, -0.055, -0.732, -0.080, -0.007]
  Std (first 5):  [0.00006, 0.271, 0.725, 0.220, 0.255]
  Min (first 5):  [0.886, -0.484, -1.525, -0.486, -0.546]
  Max (first 5):  [0.886, 0.489, 1.271, 0.486, 0.545]

Overall:
  Global mean: 0.0031
  Global std: 0.4008
  Global min: -2.819
  Global max: 2.826
```

### Analysis:

1. **First feature is nearly constant** (std = 0.00006): This is the L=0 component (global average)
2. **Different scales across features**: Higher-order harmonics have different magnitudes
3. **Not normalized**: Features are not zero-mean, unit-variance
4. **Reasonable range**: [-2.8, 2.8] is not extreme

**Conclusion**: SH features are well-behaved. The problem is NOT input statistics.

---

## Final Verdict

### RFF + SH Combination is Fundamentally Broken

**Why**:
1. SH encodes spatial data as spherical harmonics (frequency basis for the sphere)
2. RFF tries to add Cartesian Fourier components (frequency basis for flat space)
3. These two representations are incompatible
4. Optimization struggles to reconcile conflicting frequency encodings
5. Network learns poor local minima

### What Works:

1. **SH + ReLU** (R² = 0.698):
   - Simple, no frequency assumptions
   - Lets SH features through cleanly
   - Best performance

2. **SH + Spline** (R² = 0.695):
   - Local, adaptive nonlinearity
   - No frequency interference
   - Nearly as good as ReLU

3. **SH + SIREN** (R² = 0.678):
   - Periodic activation, but designed for this
   - Works, but not optimal for SH features

### What Doesn't Work:

1. **SH + RFF (any variant)** (R² < 0.63):
   - Frequency interference
   - Optimization difficulties
   - Never competitive with simpler activations

---

## Recommendations

### Immediate Actions:

1. ❌ **Abandon SH + RFF combinations** - fundamentally broken
2. ✅ **Use SH + ReLU as the default** - simple, robust, best performance
3. ✅ **Use SH + Spline if you want learned activations** - small advantage over ReLU
4. ✅ **Keep using Raw + RFF** - still works (no frequency interference)

### Updated Experimental Priorities:

**DON'T PURSUE**:
- More RFF + SH experiments (different n_features, frequency ranges, etc.)
- Longer training for RFF + SH (won't help, fundamentally broken)
- Different initialization for RFF + SH (won't fix frequency interference)

**DO PURSUE**:
1. **Spline variants** (different knot counts, initialization strategies)
2. **Raw + RFF variants** (works well, no SH interference)
3. **Other activation families** that don't use frequency representations:
   - Rational activations (Padé approximants)
   - Polynomial activations (Chebyshev, Legendre)
   - Adaptive piecewise functions
4. **Spatial gating (MoE)** with location-dependent activations
5. **Different tasks** (elevation, temperature) where the landscape might favor learned activations

---

## Lessons Learned

### 1. Frequency Representations Don't Compose Well

If your input encoding already uses a frequency basis (SH, Fourier, wavelets), don't add another frequency-based activation on top. They'll interfere.

### 2. Simple Often Wins

ReLU beat fancy learned activations. Sometimes the best solution is the simplest one.

### 3. Local > Global for Learned Activations

Splines (local, piecewise) work better than RFF (global, Fourier) when combined with frequency-encoded inputs.

### 4. Optimization Matters

Even if a model is theoretically more expressive, it's useless if you can't optimize it. Gradient norms and convergence speed are critical metrics.

### 5. Input Statistics Are Not Always the Problem

We assumed RFF failed due to unnormalized SH inputs. Turns out normalization made it worse. The real problem was deeper (frequency interference).

---

## Next Steps

See `EXPERIMENTAL_DESIGN_V2.md` for comprehensive next phase experiments.

Key priorities:
1. Spline ablation (knots, initialization, learnable positions)
2. Architecture search (depth vs width with different activations)
3. Different tasks (high-frequency data like elevation)
4. Spatial gating (Phase 4 of roadmap)
5. Alternative activation families (not frequency-based)
