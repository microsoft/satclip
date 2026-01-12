# Experimental Assumptions and Coverage

## Document Purpose

This document catalogs:
1. **All assumptions** made in experiments so far
2. **What has been tested** across notebooks 14-17
3. **What variants exist** but haven't been tested
4. **Design decisions** and their rationale
5. **Known limitations** of current experiments

---

## Table of Contents

1. [Experimental Setup Assumptions](#experimental-setup-assumptions)
2. [Data and Sampling Assumptions](#data-and-sampling-assumptions)
3. [Model Architecture Assumptions](#model-architecture-assumptions)
4. [Training Assumptions](#training-assumptions)
5. [Evaluation Assumptions](#evaluation-assumptions)
6. [What Has Been Tested](#what-has-been-tested)
7. [What Has NOT Been Tested](#what-has-not-been-tested)
8. [Known Limitations](#known-limitations)

---

## Experimental Setup Assumptions

### Hardware/Software

**Assumption**: GPU availability and CUDA compatibility
- **Rationale**: Experiments run on Google Colab with T4 GPUs
- **Impact**: Training time ~5-10 minutes per model
- **Not tested**: CPU performance, multi-GPU training, different hardware

**Assumption**: PyTorch defaults are reasonable
- **Rationale**: Standard PyTorch initialization and operations
- **Impact**: Kaiming init for ReLU, SIREN-specific init for SIREN
- **Not tested**: TensorFlow, JAX, or other frameworks

### Random Seeds

**Assumption**: Single random seed (42) is sufficient
- **Current practice**: Same seed for data sampling, not for model initialization
- **Impact**: Data split is consistent, but model initialization varies
- **Not tested**: Multiple runs with different seeds for error bars

---

## Data and Sampling Assumptions

### Dataset: GPW Population Density (2020, 15-min resolution)

**Assumption**: Population density is a good proxy task for learned activations
- **Rationale**: Smooth, low-frequency spatial phenomenon
- **Properties**:
  - Resolution: 15 arcmin (720 × 1440 grid)
  - Coverage: Global
  - Values: Log-transformed for stability
- **Not tested**: Other tasks (elevation, temperature, landcover, MOSAIKS features)

### Spatial Blocking

**Assumption**: 5° × 5° grid blocking prevents spatial leakage
- **Parameters**:
  - Block size: 5.0 degrees
  - Train/test ratio: 70/30
  - Number of blocks: 72 × 36 = 2,592 total
  - Test blocks: ~778 blocks (30%)
- **Rationale**: Prevents nearby points from being in both train and test sets
- **Impact**: More realistic evaluation than random sampling
- **Not tested**: Different block sizes (1°, 2.5°, 10°), different test ratios

### Sample Size

**Assumption**: 15,000 samples is sufficient
- **Current**: ~10,500 train, ~4,500 test
- **Rationale**: Balance between training time and data coverage
- **Not tested**: More samples (50K, 100K), fewer samples (5K, 10K)

### Data Preprocessing

**Assumption**: Log1p transformation is appropriate
- **Formula**: `y = log(1 + population_density)`
- **Rationale**: Stabilizes variance, handles zeros
- **Not tested**: Other transformations (sqrt, log10, quantile normalization)

**Assumption**: Coordinates normalized to [-1, 1] for raw inputs
- **Formula**: `lon_norm = lon / 180, lat_norm = lat / 90`
- **Rationale**: Standard practice for coordinate inputs
- **Not tested**: Other normalization schemes

---

## Model Architecture Assumptions

### Network Structure

**Assumption**: 3 hidden layers with 256 units each is optimal
- **Architecture**: `[input_dim, 256, 256, 256, 256]`
- **Rationale**: Matches SatCLIP architecture for fair comparison
- **Parameter counts**:
  - Raw (2D) input: ~231K params
  - SH(L=10) input: ~256K params
- **Not tested**: Different depths (2, 4, 5, 8 layers), different widths (128, 512, 1024)

### Output Head

**Assumption**: Simple 2-layer MLP head is sufficient
- **Structure**: `[256 → 128 → 1]` with ReLU in between
- **Parameters**: ~33K params
- **Rationale**: Lightweight, focuses evaluation on encoder
- **Not tested**: Different head architectures, direct linear projection

### Input Encodings

#### Raw Coordinates (2D)

**Assumptions**:
- Coordinates are sufficient without additional features
- Normalization to [-1, 1] is appropriate

**Properties**:
- Input dim: 2 (lon, lat)
- No learnable parameters in encoding
- Simple, minimal

**Not tested**: Fourier features, positional encoding variants

#### Spherical Harmonics (SH)

**Assumptions**:
- SH(L=10) with 100 features is a good default
- Analytic computation is correct
- No normalization needed (we tested this in NB17!)

**Properties**:
- L=10 → 100 features (not 121 = (L+1)²)
- Frequency content: captures scales from global down to ~1000km
- No learnable parameters

**Not tested**:
- Different L values (5, 15, 20, 40)
- Other spherical encodings (Zernike, spherical CNNs)

---

## Training Assumptions

### Optimization

**Assumption**: Adam with lr=1e-3 is a good default
- **Optimizer**: Adam with default betas (0.9, 0.999)
- **Learning rate**: 1e-3 (fixed, no schedule)
- **Rationale**: Standard choice for MLPs, usually works well
- **Not tested**:
  - Different optimizers (SGD, AdamW, RMSprop)
  - Different learning rates (1e-4, 1e-2, 5e-3)
  - Learning rate schedules (cosine, step decay, plateau)

**Assumption**: 100 epochs is sufficient for convergence
- **Observation**: ReLU/Spline converge in ~20-30 epochs
- **Issue**: RFF appears to need more time but still underperforms
- **Not tested**:
  - Longer training (200, 500, 1000 epochs)
  - Early stopping based on validation performance

### Batch Size

**Assumption**: Batch size 256 is appropriate
- **Rationale**: Balances GPU memory and gradient noise
- **Training samples**: ~10,500 → ~41 batches per epoch
- **Not tested**: Different batch sizes (64, 128, 512)

### Loss Function

**Assumption**: MSE loss is appropriate for regression
- **Formula**: `MSE = mean((y_pred - y_true)²)`
- **Rationale**: Standard for regression, easy to optimize
- **Not tested**:
  - Other losses (MAE, Huber, quantile loss)
  - Weighted losses (spatial weighting, uncertainty weighting)

### Regularization

**Assumption**: No explicit regularization is needed
- **Current**: No weight decay, no dropout
- **Rationale**: Models are relatively small, not obviously overfitting
- **Observation**: Training and test loss both decrease
- **Not tested**:
  - Weight decay (1e-4, 1e-5)
  - Dropout (0.1, 0.2)
  - Batch normalization, layer normalization

---

## Evaluation Assumptions

### Metric: R² Score

**Assumption**: R² is the primary metric of interest
- **Formula**: `R² = 1 - SS_res / SS_tot`
- **Rationale**: Interpretable, standard for regression
- **Range**: (-∞, 1], with 1 being perfect
- **Not tested**: Other metrics (RMSE, MAE, correlation)

### Efficiency Metric

**Assumption**: R² per 10K parameters measures parameter efficiency
- **Formula**: `Efficiency = R² / (params / 10000)`
- **Rationale**: Favors models that achieve high R² with fewer parameters
- **Not tested**: FLOPs-based efficiency, inference time efficiency

### Test Set Evaluation

**Assumption**: Single test set evaluation is sufficient
- **Current**: Evaluate on held-out test set, report final R²
- **Not tested**:
  - Cross-validation (k-fold)
  - Multiple test sets (different regions, different years)
  - Bootstrap confidence intervals

---

## Activation Function Assumptions

### RFF (Random Fourier Features)

**Design Choices**:
- `n_features`: Number of sin/cos pairs (default: 25)
- `max_freq`: Maximum frequency (default: 10.0)
- `learnable_freq`: Whether frequencies are learnable (default: False)
- `freq_init`: Initialization method ('linear', 'log', 'random')

**Assumptions**:
- Linear frequency spacing [0.1, 10] is appropriate
- 25 features is a good default
- Fixed frequencies work better than learnable
- Each neuron gets its own independent RFF activation

**What we tested** (Notebooks 14-17):
- ✅ n_features: 10, 25, 50, 100
- ✅ learnable_freq: True vs False (NB17)
- ✅ With raw coordinates (works)
- ✅ With SH features (fails)
- ✅ With vs without input normalization (NB17)

**What we haven't tested**:
- ❌ freq_init: 'log', 'random'
- ❌ Different max_freq: 5, 20, 50
- ❌ Shared vs independent RFF across neurons
- ❌ RFF at only some layers (not all)

### Spline (Piecewise Linear)

**Design Choices**:
- `n_knots`: Number of control points (default: 10)
- `input_range`: Range of input values (default: (-3, 3))
- `init`: Initialization method (default: 'relu')

**Assumptions**:
- 10 knots is sufficient for most functions
- Input range [-3, 3] covers ~99.7% of normalized activations
- ReLU initialization provides a good starting point
- Linear interpolation is sufficient (vs cubic splines)

**What we tested** (Notebooks 16-17):
- ✅ n_knots: 10, 30
- ✅ With raw coordinates (works)
- ✅ With SH features (works well)

**What we haven't tested**:
- ❌ n_knots: 5, 15, 20, 50, 100
- ❌ init: 'linear', 'zero', 'random'
- ❌ input_range: (-5, 5), (-10, 10)
- ❌ Learnable knot positions
- ❌ Cubic spline interpolation
- ❌ Monotonic constraints

### SIREN

**Design Choices**:
- `w0_initial`: Frequency scaling for first layer (30.0)
- `w0_hidden`: Frequency scaling for hidden layers (1.0)
- Special initialization based on w0

**Assumptions**:
- SIREN initialization is critical (from original paper)
- w0 = 30 for first layer, w0 = 1 for hidden is optimal
- Sine activation without any learnable parameters

**What we tested** (Notebooks 14-17):
- ✅ With raw coordinates (works well)
- ✅ With SH features (works, but not best)

**What we haven't tested**:
- ❌ Different w0 values
- ❌ Learnable w0 (per layer or per neuron)
- ❌ Other periodic functions (cos, tanh)

### ReLU

**Assumptions**:
- Standard Kaiming initialization is optimal
- No hyperparameters to tune

**What we tested** (Notebooks 16-17):
- ✅ With SH features (works best!)

**What we haven't tested**:
- Nothing - ReLU is fully specified

---

## What Has Been Tested

### Notebook 14 (Initial CPU Tests)
- ✅ RFF (n=25) vs Spline (k=10) vs ReLU on raw coordinates
- ✅ 15-min resolution, CPU training
- ✅ 100 epochs, lr=1e-3

### Notebook 15 (Multi-Resolution GPU)
- ✅ RFF (n=25) vs Spline (k=10) vs ReLU on raw coordinates
- ✅ Three resolutions: 15-min, 30-min, 1-degree
- ✅ Comparison to SatCLIP L=10 and L=40
- ✅ GPU training, 100 epochs

### Notebook 16 (Phase 1 Core 2×2)
- ✅ Raw vs SH(L=10) input encoding
- ✅ SIREN vs RFF vs Spline vs ReLU activations
- ✅ RFF variants: n=25, 50, 100
- ✅ Spline variants: k=10, 30
- ✅ 12 model combinations total

### Notebook 17 (Diagnostic)
- ✅ SH feature statistics and distributions
- ✅ Input normalization for SH features
- ✅ Learnable vs fixed frequencies for RFF
- ✅ Training dynamics: loss curves, gradient norms, R² progression
- ✅ 6 diagnostic experiments

**Total experiments across notebooks: 41+**

---

## What Has NOT Been Tested

### Input Encodings
- ❌ SH with different L values (5, 15, 20, 40)
- ❌ Fourier features (standard, Gaussian)
- ❌ Wavelets
- ❌ Polynomial features
- ❌ Grid-based features (binned coordinates)

### Activation Variants
- ❌ RFF: log/random frequency initialization
- ❌ RFF: different max_freq values (5, 20, 50)
- ❌ RFF: shared frequencies across neurons
- ❌ Spline: different knot counts (5, 15, 20, 50, 100)
- ❌ Spline: different initializations (linear, zero, random)
- ❌ Spline: learnable knot positions
- ❌ Spline: cubic interpolation
- ❌ SIREN: different w0 values
- ❌ Other activations: GELU, Swish, Mish, PReLU, ELU

### Architecture Variants
- ❌ Different depths: 2, 4, 5, 8 layers
- ❌ Different widths: 128, 512, 1024
- ❌ Residual connections
- ❌ Skip connections
- ❌ Different output heads
- ❌ Layer normalization / batch normalization

### Training Variants
- ❌ Different optimizers: SGD, AdamW, RMSprop
- ❌ Different learning rates: 1e-4, 5e-3, 1e-2
- ❌ Learning rate schedules: cosine, step, plateau
- ❌ Longer training: 200, 500, 1000 epochs
- ❌ Different batch sizes: 64, 128, 512
- ❌ Regularization: weight decay, dropout
- ❌ Early stopping

### Data Variants
- ❌ Different sample sizes: 5K, 10K, 50K, 100K
- ❌ Different spatial blocking sizes: 1°, 2.5°, 10°
- ❌ Different train/test ratios: 80/20, 60/40
- ❌ Different data transformations: sqrt, log10
- ❌ Multiple random seeds for error bars

### Task Variants
- ❌ Elevation (high-frequency, sharp features)
- ❌ Temperature (medium-frequency, smooth)
- ❌ Landcover classification
- ❌ MOSAIKS features (multiple tasks)
- ❌ Urban density (discrete boundaries)

### Advanced Techniques
- ❌ Spatial gating / Mixture of Experts (Phase 4)
- ❌ Location-dependent activations
- ❌ Ensemble methods
- ❌ Transfer learning from SatCLIP
- ❌ Contrastive training (Phase 5)

---

## Known Limitations

### 1. Single Random Seed
- **Issue**: No error bars, results could be noisy
- **Impact**: Can't quantify uncertainty
- **Solution**: Run 3-5 times with different seeds

### 2. Fixed 100 Epochs
- **Issue**: RFF might need more time, ReLU might be done earlier
- **Impact**: Potentially unfair comparison
- **Solution**: Use early stopping or train to convergence

### 3. No Hyperparameter Tuning
- **Issue**: Same learning rate, batch size for all models
- **Impact**: Some models might benefit from different settings
- **Solution**: Per-model grid search

### 4. Single Task
- **Issue**: Only tested on population density
- **Impact**: Conclusions might not generalize
- **Solution**: Test on elevation, temperature, etc.

### 5. Single Resolution
- **Issue**: Mostly tested at 15-min (NB15 had 3 resolutions)
- **Impact**: Don't know how activations scale with resolution
- **Solution**: Systematic multi-resolution study

### 6. No Architecture Search
- **Issue**: Fixed 3×256 architecture
- **Impact**: Don't know optimal depth/width for each activation
- **Solution**: Depth-width sweep for each activation

### 7. No Statistical Testing
- **Issue**: R² differences treated as deterministic
- **Impact**: Can't say if differences are significant
- **Solution**: Multiple runs + t-tests / bootstrap

### 8. GPU-Only Testing
- **Issue**: Don't know CPU performance or inference time
- **Impact**: Can't optimize for deployment
- **Solution**: Benchmark inference speed

### 9. No Visualization of Learned Activations
- **Issue**: Don't know what shapes RFF/Spline learn
- **Impact**: Can't debug or interpret learned functions
- **Solution**: Plot activation functions after training

### 10. No Ablation of Activation Position
- **Issue**: Always apply activation at every layer
- **Impact**: Don't know if activations at all layers are necessary
- **Solution**: Test activations at only some layers

---

## Design Decisions and Rationale

### Why This Architecture?

**Decision**: 3 layers × 256 hidden units

**Rationale**:
1. Matches SatCLIP encoder for fair comparison
2. ~230-256K params is reasonable for this dataset size
3. Not too deep (hard to train) or too shallow (limited expressiveness)

**Trade-offs**:
- Deeper might be more expressive but harder to optimize
- Wider might fit better but risk overfitting

### Why These Activations?

**Decision**: RFF, Spline, SIREN, ReLU

**Rationale**:
1. **RFF**: Explicit frequency representation, theoretically principled
2. **Spline**: Local, adaptive, piecewise linear (interpretable)
3. **SIREN**: Prior work baseline (SatCLIP uses it)
4. **ReLU**: Simple baseline, widely used

**Alternatives considered but not tested**:
- GELU, Swish, Mish (modern alternatives to ReLU)
- Rational functions (Padé approximants)
- Wavelets (localized frequency representation)

### Why Population Density?

**Decision**: GPW population density dataset

**Rationale**:
1. **Available**: Free, public dataset
2. **Global**: Covers all regions, diverse spatial patterns
3. **Smooth**: Low-frequency task, good for initial testing
4. **Prior work**: SatCLIP was evaluated on this task

**Limitations**:
- Too smooth? Might not benefit from learned activations
- Need high-frequency tasks (elevation) to stress-test

### Why 100 Epochs?

**Decision**: Fixed 100 epochs for all models

**Rationale**:
1. **Reasonable compute**: ~5-10 min per model on GPU
2. **ReLU converges**: Simple models finish in 20-30 epochs
3. **Consistency**: Same budget for all models

**Issues**:
- RFF might need 500+ epochs (but still underperforms)
- Early stopping would be more fair

---

## Summary

### Comprehensive Testing:
- ✅ **4 notebooks, 40+ experiments**
- ✅ **Multiple input encodings** (raw, SH)
- ✅ **Multiple activations** (RFF, Spline, SIREN, ReLU)
- ✅ **Diagnostic analysis** (normalization, gradients, dynamics)

### Key Findings:
1. **RFF + SH doesn't work** (frequency interference confirmed)
2. **Spline + SH works well** (+2.5% vs SIREN)
3. **ReLU + SH works best** (+2.9% vs SIREN)
4. **Normalization doesn't fix RFF** (made it worse)
5. **Learnable frequencies don't help** (made it much worse)

### Remaining Unknowns:
1. Do learned activations help on **high-frequency tasks**?
2. What is the **optimal architecture** for each activation?
3. Do results hold with **multiple random seeds**?
4. Can **longer training** improve RFF? (probably not)
5. What do **learned activation shapes** look like?

### Next Phase Priorities:
1. **Spline ablation** (knots, init, architecture)
2. **High-frequency tasks** (elevation, edges)
3. **Architecture search** (depth vs width)
4. **Visualization** (plot learned functions)
5. **Robustness** (multiple seeds, error bars)

See `EXPERIMENTAL_DESIGN_V2.md` for detailed next steps.
