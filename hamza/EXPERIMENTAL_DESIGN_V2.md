# Experimental Design V2: Comprehensive Learned Activation Analysis

## Executive Summary

Based on notebooks 14-17, we've learned:
1. **RFF + SH is fundamentally broken** (frequency interference)
2. **Spline + SH works well** (+2.5% vs SIREN, stable)
3. **ReLU + SH is the winner** (+2.9% vs SIREN, simplest)
4. **Raw + RFF still works** (no SH interference)

**Next phase goal**: Comprehensive characterization of when, where, and why learned activations help (or don't).

---

## Design Philosophy

### What We're NOT Doing:
❌ Testing 100 random hyperparameter combinations
❌ Grid searching every possible RFF configuration
❌ Trying to "fix" RFF + SH (it's fundamentally broken)

### What We ARE Doing:
✅ **Systematic characterization** of learned activation properties
✅ **Task-based analysis** (when do learned activations help?)
✅ **Architecture interaction** (how do activations interact with depth/width?)
✅ **Visualization and interpretation** (what are they learning?)
✅ **Robustness analysis** (do results hold across seeds/splits?)

---

## Phase 2: Comprehensive Analysis Framework

### Notebook 18: Spline Deep Dive

**Goal**: Comprehensive characterization of spline activations

**Why Spline?**
- Works well with SH (+2.5% vs SIREN)
- Works well with raw coordinates
- Local, interpretable, stable gradients
- More promising than RFF for learned activations

#### Experiment 1: Spline Capacity Analysis

**Test different knot counts to find optimal expressiveness**

```python
spline_configs = [
    {'n_knots': 5,   'name': 'Minimal'},
    {'n_knots': 10,  'name': 'Default'},
    {'n_knots': 15,  'name': 'Medium'},
    {'n_knots': 20,  'name': 'High'},
    {'n_knots': 30,  'name': 'Very High'},
    {'n_knots': 50,  'name': 'Extreme'},
]

# Test with both raw and SH inputs
inputs = ['raw', 'sh_L10']

# Track: R², params, training time, final activation shapes
```

**Expected outcome**: Identify optimal knot count for different input types

**Analysis**:
- Plot R² vs n_knots
- Plot overfitting (train-test gap) vs n_knots
- Visualize learned activation shapes for each
- Check if more knots always help or if there's a sweet spot

#### Experiment 2: Spline Initialization Strategies

**Test how initialization affects convergence and final performance**

```python
spline_inits = [
    {'init': 'relu',   'name': 'ReLU-like'},
    {'init': 'linear', 'name': 'Identity'},
    {'init': 'zero',   'name': 'Zero'},
    {'init': 'tanh',   'name': 'Tanh-like'},  # NEW
    {'init': 'gelu',   'name': 'GELU-like'},  # NEW
]

# Fix n_knots=10 for fair comparison
# Test with SH(L=10) input
```

**Expected outcome**: Determine if initialization matters

**Analysis**:
- Compare convergence speed (epochs to reach 90% of final R²)
- Plot learning curves for each initialization
- Visualize final activation shapes
- Check if they all converge to similar shapes

#### Experiment 3: Spline Input Range Sensitivity

**Test if input range affects performance**

```python
spline_ranges = [
    {'input_range': (-3, 3),   'name': '99.7% coverage'},
    {'input_range': (-5, 5),   'name': '99.99% coverage'},
    {'input_range': (-10, 10), 'name': 'Very wide'},
]

# Fix n_knots=10, init='relu'
# Test with SH(L=10) input
```

**Expected outcome**: Understand if clamping at boundaries is an issue

**Analysis**:
- Measure how often activations hit the boundaries
- Plot activation distribution vs input range
- Check if wider range improves performance

#### Experiment 4: Learnable Knot Positions

**Test if learning knot positions helps**

```python
spline_learnable = [
    {'learnable_positions': False, 'name': 'Fixed positions'},
    {'learnable_positions': True,  'name': 'Learnable positions'},
]

# Fix n_knots=10, init='relu', range=(-3,3)
# Test with SH(L=10) input
```

**Expected outcome**: Determine if adaptive knot spacing helps

**Analysis**:
- Compare learned knot positions to uniform spacing
- Check if knots cluster in certain regions
- Measure improvement over fixed positions

#### Experiment 5: Spline Interpolation Methods

**Test different interpolation schemes**

```python
spline_interp = [
    {'method': 'linear', 'name': 'Piecewise Linear'},
    {'method': 'cubic',  'name': 'Cubic Spline'},    # NEW
]

# Fix n_knots=10, init='relu'
# Test with SH(L=10) input
```

**Expected outcome**: Determine if smoothness matters

**Analysis**:
- Compare R², convergence speed
- Visualize activation shapes (are cubic smoother?)
- Check gradient norms (are cubic more stable?)

---

### Notebook 19: Architecture Interaction Study

**Goal**: Understand how learned activations interact with network architecture

**Why?** Maybe learned activations shine with different architectures than ReLU

#### Experiment 1: Depth Sweep

**Test different depths with different activations**

```python
depths = [2, 3, 4, 5, 8]
activations = ['relu', 'spline', 'siren']

# Fix width=256, SH(L=10) input
# For each (depth, activation) pair, measure:
#   - R² performance
#   - Training time
#   - Gradient norms
#   - Convergence speed
```

**Expected outcome**: Find optimal depth for each activation

**Analysis**:
- Plot R² vs depth for each activation
- Identify crossover points (where Spline > ReLU)
- Check if learned activations help with deeper networks

#### Experiment 2: Width Sweep

**Test different widths with different activations**

```python
widths = [128, 256, 384, 512]
activations = ['relu', 'spline', 'siren']

# Fix depth=3, SH(L=10) input
```

**Expected outcome**: Find optimal width for each activation

**Analysis**:
- Plot R² vs width for each activation
- Control for parameter count (normalize by efficiency)
- Check if learned activations help with narrower networks

#### Experiment 3: Depth-Width Trade-off

**Match parameter count, vary architecture**

```python
configs = [
    {'depth': 2, 'width': 362},  # ~256K params
    {'depth': 3, 'width': 256},  # ~256K params
    {'depth': 4, 'width': 215},  # ~256K params
    {'depth': 5, 'width': 192},  # ~256K params
]

activations = ['relu', 'spline']
```

**Expected outcome**: Determine if depth or width matters more

**Analysis**:
- Control for parameter count
- Identify architecture preferences per activation
- Test hypothesis: "Spline works better with deeper networks"

---

### Notebook 20: Task-Based Analysis

**Goal**: Identify which tasks benefit from learned activations

**Why?** Population density might be too smooth - need high-frequency tasks

#### Experiment 1: Elevation (High-Frequency)

**Test on ETOPO1 elevation data**

```python
# Load elevation data (sharp peaks, valleys)
# Sample with spatial blocking
# Train: ReLU vs Spline vs SIREN

# Expected: Learned activations might help here
```

**Properties**:
- High-frequency features (mountains, valleys)
- Sharp discontinuities
- Fractal-like structures

**Analysis**:
- Compare to population density results
- Check if Spline > ReLU for elevation
- Visualize predictions on complex terrain

#### Experiment 2: Temperature Anomalies (Medium-Frequency)

**Test on climate reanalysis temperature data**

```python
# Load temperature anomaly data
# Sharp fronts, medium-frequency variations

# Expected: Medium difficulty, between population and elevation
```

**Properties**:
- Medium-frequency variations
- Some sharp fronts
- Smoother than elevation, rougher than population

#### Experiment 3: Urban/Rural Boundaries (Discrete)

**Test on binary urban classification**

```python
# Derived from population density
# Threshold at high density to get urban/rural labels
# Binary classification task

# Expected: Edges matter, might favor nonlinear activations
```

**Properties**:
- Discrete boundaries
- Step function-like transitions
- Tests ability to represent sharp transitions

#### Experiment 4: Multi-Task Learning

**Train single encoder for all 3 tasks**

```python
# Shared encoder with task-specific heads
# Test if learned activations generalize better

# Expected: Spline might be more versatile than ReLU
```

**Analysis**:
- Average performance across tasks
- Check if Spline is more robust
- Identify task-specific strengths

---

### Notebook 21: Visualization and Interpretation

**Goal**: Understand what learned activations are learning

**Why?** Black box testing isn't enough - need to interpret

#### Experiment 1: Activation Shape Visualization

**Plot learned activation functions**

```python
# After training, extract spline knot values
# Plot g(x) for each layer

# For RFF: plot sum of sin/cos components
# For Spline: plot piecewise linear function
```

**Analysis**:
- Do different layers learn different shapes?
- Do activations look like ReLU, GELU, or something new?
- Are shapes consistent across training runs?

#### Experiment 2: Layer-wise Activation Analysis

**Measure activation statistics per layer**

```python
# During forward pass, log:
#   - Mean, std, min, max of activations
#   - Sparsity (% of near-zero activations)
#   - Gradient magnitudes

# Compare ReLU vs Spline layer by layer
```

**Analysis**:
- Do learned activations maintain better gradient flow?
- Are deeper layers different from shallow layers?
- Check for dead neurons or saturation

#### Experiment 3: Input-Output Mapping

**Visualize what each activation does to inputs**

```python
# Sample range of inputs x ∈ [-5, 5]
# Plot g(x) for trained activation
# Color by layer or neuron

# Compare to idealized ReLU, GELU, etc.
```

**Analysis**:
- Do activations approximate known functions?
- Are there qualitative differences from standard activations?
- Can we interpret the learned shapes?

#### Experiment 4: Ablation by Layer

**Test if activations at all layers are necessary**

```python
configs = [
    {'activation_layers': [0, 1, 2],    'name': 'All layers'},
    {'activation_layers': [0],          'name': 'First only'},
    {'activation_layers': [0, 1],       'name': 'First two'},
    {'activation_layers': [2],          'name': 'Last only'},
]

# Use ReLU at non-activation layers
# Use Spline at activation layers
```

**Analysis**:
- Are all layers equally important?
- Does first layer matter most (input encoding)?
- Does last layer matter most (output mapping)?

---

### Notebook 22: Robustness and Generalization

**Goal**: Verify that results are robust and not due to lucky seeds

**Why?** So far, only 1 run per config - need error bars

#### Experiment 1: Multiple Random Seeds

**Run key experiments with 5 different seeds**

```python
seeds = [42, 123, 456, 789, 2024]

key_configs = [
    {'input': 'sh_L10', 'activation': 'relu'},
    {'input': 'sh_L10', 'activation': 'spline', 'n_knots': 10},
    {'input': 'sh_L10', 'activation': 'siren'},
]

# For each config, train with all seeds
# Report: mean ± std of R²
```

**Analysis**:
- Compute confidence intervals
- Test statistical significance (t-test)
- Check if Spline > ReLU is significant

#### Experiment 2: Different Spatial Blocking

**Test with different block sizes**

```python
block_sizes = [2.5, 5.0, 10.0]  # degrees

# Affects train/test split
# Smaller blocks = more similar train/test
# Larger blocks = more different train/test
```

**Analysis**:
- Check if results hold across different splits
- Measure generalization difficulty vs block size
- See if learned activations help more with harder splits

#### Experiment 3: Different Train/Test Ratios

**Test with different data amounts**

```python
train_ratios = [0.5, 0.7, 0.9]

# 50% train = harder (less data)
# 90% train = easier (more data)
```

**Analysis**:
- Do learned activations help more with less data?
- Test hypothesis: "Spline regularizes better than ReLU"

#### Experiment 4: Transfer Learning

**Train on one region, test on another**

```python
# Train on Northern Hemisphere, test on Southern
# Train on land, test on ocean-adjacent regions
```

**Analysis**:
- Measure generalization to different regions
- Check if learned activations transfer better
- Identify failure modes

---

### Notebook 23: Raw + Learned Activation Deep Dive

**Goal**: Understand why Raw + RFF works but SH + RFF doesn't

**Why?** Raw + RFF was competitive with SH + SIREN in NB15

#### Experiment 1: Raw + RFF Parameter Sweep

**Comprehensive RFF tuning for raw coordinates**

```python
rff_configs = [
    # n_features sweep
    {'n_features': 10,  'freq_init': 'linear', 'max_freq': 10},
    {'n_features': 25,  'freq_init': 'linear', 'max_freq': 10},
    {'n_features': 50,  'freq_init': 'linear', 'max_freq': 10},
    {'n_features': 100, 'freq_init': 'linear', 'max_freq': 10},

    # freq_init sweep
    {'n_features': 50, 'freq_init': 'linear', 'max_freq': 10},
    {'n_features': 50, 'freq_init': 'log',    'max_freq': 10},
    {'n_features': 50, 'freq_init': 'random', 'max_freq': 10},

    # max_freq sweep
    {'n_features': 50, 'freq_init': 'linear', 'max_freq': 5},
    {'n_features': 50, 'freq_init': 'linear', 'max_freq': 20},
    {'n_features': 50, 'freq_init': 'linear', 'max_freq': 50},
]

# All with raw coordinate input
```

**Expected outcome**: Find optimal RFF configuration for raw inputs

**Analysis**:
- Compare to Raw + SIREN baseline
- Check if RFF can match or beat SIREN
- Identify frequency range that captures spatial scales

#### Experiment 2: Raw + Spline vs Raw + SIREN

**Head-to-head comparison**

```python
configs = [
    {'input': 'raw', 'activation': 'siren'},
    {'input': 'raw', 'activation': 'spline', 'n_knots': 10},
    {'input': 'raw', 'activation': 'spline', 'n_knots': 20},
    {'input': 'raw', 'activation': 'spline', 'n_knots': 30},
]
```

**Expected outcome**: Can Spline beat SIREN without SH?

**Analysis**:
- Test if Spline can discover frequencies
- Compare to SH + Spline performance
- Check if adding SH features helps Spline more than SIREN

#### Experiment 3: Hybrid Approaches

**Combine different encodings/activations**

```python
configs = [
    # Raw + SH concatenation
    {'input': 'raw+sh_L10', 'activation': 'relu'},
    {'input': 'raw+sh_L10', 'activation': 'spline'},

    # Different activations per layer
    {'layer_activations': ['rff', 'spline', 'relu']},
    {'layer_activations': ['siren', 'spline', 'spline']},
]
```

**Expected outcome**: Find best of both worlds

**Analysis**:
- Does concatenating raw + SH help?
- Do hybrid activation schemes work?
- Identify complementary combinations

---

## Experimental Priorities

### High Priority (Do First)

1. **Notebook 18: Spline Deep Dive** (Experiment 1 & 2)
   - Most promising learned activation
   - Need to understand capacity and initialization

2. **Notebook 21: Visualization** (Experiment 1 & 3)
   - Critical for interpretation
   - Can run alongside Notebook 18

3. **Notebook 22: Robustness** (Experiment 1)
   - Need error bars for key results
   - Required for publication/claims

### Medium Priority (Do Next)

4. **Notebook 20: Task-Based Analysis** (Experiment 1)
   - Elevation is high-frequency, critical test
   - Will determine if learned activations are useful

5. **Notebook 19: Architecture Interaction** (Experiment 1)
   - Depth sweep is most important
   - Might reveal where learned activations shine

6. **Notebook 23: Raw + Learned** (Experiment 1 & 2)
   - Complete the Raw vs SH story
   - Test if RFF can work without SH

### Lower Priority (Nice to Have)

7. Remaining experiments in Notebooks 18-23
8. Multi-task learning (Notebook 20, Experiment 4)
9. Transfer learning (Notebook 22, Experiment 4)
10. Hybrid approaches (Notebook 23, Experiment 3)

---

## Success Criteria

### Phase 2 is successful if we can answer:

1. **When do learned activations help?**
   - Which tasks? (smooth vs rough)
   - Which architectures? (deep vs shallow, wide vs narrow)
   - Which input encodings? (raw vs SH vs hybrid)

2. **What are learned activations learning?**
   - What shapes do splines converge to?
   - Are they approximating known functions (GELU, Swish)?
   - Do different layers learn different shapes?

3. **Are results robust?**
   - Do they hold across multiple seeds?
   - Do they generalize to different regions?
   - Are improvements statistically significant?

4. **What's the best configuration?**
   - Optimal spline knot count?
   - Optimal architecture for splines?
   - Best input encoding?

---

## Estimated Timeline

**Assuming GPU access (Colab T4)**:

- **Notebook 18**: ~2-3 hours (5 experiments × 20-30 models)
- **Notebook 19**: ~1-2 hours (depth/width sweeps)
- **Notebook 20**: ~3-4 hours (new datasets, multiple tasks)
- **Notebook 21**: ~1 hour (visualization, no training)
- **Notebook 22**: ~2-3 hours (5 seeds × key configs)
- **Notebook 23**: ~2-3 hours (RFF parameter sweep)

**Total**: ~12-16 hours of compute

**Calendar time**: 3-4 days (running multiple notebooks in parallel)

---

## Deliverables

### After Phase 2, we will have:

1. **Comprehensive Spline Characterization**
   - Optimal knot count, initialization, architecture
   - Visualization of learned activation shapes
   - Guidelines for when to use splines

2. **Task-Based Understanding**
   - Know which tasks benefit from learned activations
   - Elevation results (critical test)
   - Multi-task performance

3. **Robustness Validation**
   - Error bars for key results
   - Statistical significance tests
   - Generalization analysis

4. **Actionable Recommendations**
   - Use Spline (k=X) for task Y with architecture Z
   - Use ReLU for task A (simpler is better)
   - Use SIREN for task B (frequency matters)

5. **Publication-Ready Results**
   - Complete story: when, where, why learned activations help
   - Visualizations and interpretations
   - Statistically validated conclusions

---

## Open Questions for Phase 3+

After Phase 2, we might want to explore:

1. **Spatial Gating (MoE)**: Location-dependent activations
2. **Contrastive Training**: Scale to dual-modality learning
3. **Other Activation Families**: Rational functions, wavelets
4. **Deployment**: Inference speed, quantization, edge devices
5. **Theory**: Why do splines work? Approximation theory analysis

But first, let's complete Phase 2 and understand the basics.

---

## Conclusion

Phase 2 shifts from "testing random configurations" to "systematic characterization". We're not trying to find the magic hyperparameters that make RFF work - we're trying to understand the fundamental properties of learned activations and when they're useful.

**Key philosophy**:
- ✅ Understand > Optimize
- ✅ Interpret > Black box test
- ✅ Robustness > Single best result
- ✅ Task-based > Configuration-based

This will produce a comprehensive understanding of learned activations that's publication-ready and actionable for practitioners.
