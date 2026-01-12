# Learned Activations Architecture Setup

This document describes the architecture setup for the learned activation experiments, including how they compare to and align with SatCLIP.

## Overview

We test whether **learned activation functions** can replace the spherical harmonics + SIREN approach used in SatCLIP for location encoding.

---

## 1. SatCLIP Baseline Architecture

### 1.1 Positional Encoding: Spherical Harmonics

SatCLIP uses spherical harmonics as positional encoding:
- **L=10**: 100 features (legendre_polys=10)
- **L=40**: 1600 features (legendre_polys=40)

The spherical harmonics transform (lon, lat) coordinates into a high-dimensional feature space that captures spatial frequency information.

### 1.2 Neural Network: SIREN

SatCLIP uses a SIREN (Sinusoidal Representation Network) to process SH features:

```
Input (SH features) -> SIREN layers -> 256-dim embedding
```

**Critical SIREN configuration** (from `satclip/location_encoder.py`):
- **First layer**: `w0_initial = 30.0`
- **Subsequent layers**: `w0 = 1.0`
- **Initialization**:
  - First layer: `uniform(-1/dim_in, 1/dim_in)`
  - Subsequent: `uniform(-sqrt(6/dim_in)/w0, sqrt(6/dim_in)/w0)`

### 1.3 Evaluation Baseline

For fair comparison, we use **Ridge regression** on frozen SatCLIP embeddings (not an MLP head which overfits).

---

## 2. Our Learned Activation Architectures

### 2.1 LearnedActivation

Fourier-parameterized activation function:

```
g(x) = scale * (sum_k a_k*sin(w_k*x) + b_k*cos(w_k*x)) + bias
```

**Parameters:**
- `n_frequencies`: Number of Fourier components K (default: 25)
- `freq_init`: 'linear', 'log', or 'random' frequency initialization
- `learnable_freq`: Whether frequencies are learnable (default: False)
- `max_freq`: Maximum frequency value (default: 10.0)

**Learnable parameters per activation:**
- `sin_coeffs`: K coefficients for sine terms
- `cos_coeffs`: K coefficients for cosine terms
- `scale`: Output scaling
- `bias`: Output bias

### 2.2 SpatiallyVaryingActivation

Mixture of expert activations with location-based gating:

```
g(x; lon, lat) = sum_m w_m(lon, lat) * expert_m(x)
```

**Key design: SHARED gating network**

The gating network is shared across ALL layers (not per-layer):

```python
# Shared gate (created once)
self.shared_gate = SharedGatingNetwork(n_experts=8, hidden_dim=64)

# Each layer uses the same gate
self.activations = [
    SpatiallyVaryingActivation(self.shared_gate, n_experts=8)
    for _ in range(n_layers)
]
```

**Gating network architecture:**
```
(lon, lat) -> Linear(2, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, n_experts) -> Softmax
```

### 2.3 Location Encoder Variants

| Encoder | Input | Activation | Init |
|---------|-------|------------|------|
| `LocationEncoderReLU` | Direct (lon, lat) | ReLU | Kaiming |
| `LocationEncoderSIREN` | Direct (lon, lat) | Sine (w0_init=30, w0=1) | SIREN |
| `LocationEncoderLearned` | Direct (lon, lat) | LearnedActivation | Kaiming |
| `LocationEncoderSpatial` | Direct (lon, lat) | SpatiallyVarying (shared gate) | Kaiming |

### 2.4 HybridEncoder

Uses SatCLIP's spherical harmonics but replaces SIREN with learned activations:

```
(lon, lat) -> SH(L=10 or L=40) -> Custom MLP -> 256-dim embedding
```

**Activation options:**
- `relu`: Standard ReLU with Kaiming init
- `siren`: SIREN with proper init (w0_init=30, w0=1)
- `learned`: LearnedActivation with Kaiming init

---

## 3. Training Setup

### 3.1 Task: Population Density Prediction

- **Dataset**: GPW v4 Population Density (15-min resolution)
- **Target**: log1p(population_density)
- **Metric**: R² score

### 3.2 Spatial Blocking (prevents leakage)

To prevent spatial autocorrelation from inflating R² scores, we use **grid-based spatial blocking**:

```python
def sample_with_spatial_blocking(data, coords, grid_size=5.0, test_ratio=0.3):
    """
    1. Divide region into grid_size x grid_size degree cells
    2. Randomly assign 30% of cells to test set
    3. All points in a cell go to same set (train or test)
    """
```

This ensures train and test points are spatially separated by at least one grid cell (~500km at equator for 5° grid).

### 3.3 Training Configuration

```python
N_SAMPLES = 20000      # Total samples
EPOCHS = 100           # Training epochs
BATCH_SIZE = 256       # Batch size
LR = 1e-3             # Learning rate (Adam)
GRID_SIZE = 5.0       # Spatial blocking grid (degrees)
TEST_RATIO = 0.3      # Fraction of grid cells for testing
```

### 3.4 Prediction Head

All encoders use the same prediction head:
```
Encoder(256) -> Linear(256, 128) -> ReLU -> Linear(128, 1)
```

---

## 4. Parameter Counts

| Model | Parameters |
|-------|------------|
| LocationEncoderReLU | ~198K |
| LocationEncoderSIREN | ~198K |
| LocationEncoderLearned | ~198K + 156/layer |
| LocationEncoderSpatial | ~198K + shared_gate(~9K) + experts(~1.2K/layer) |
| SatCLIP L=10 head (Ridge) | 256 * 1 = 256 |
| SatCLIP L=40 head (Ridge) | 256 * 1 = 256 |

---

## 5. Key Fixes from v1/v2

### 5.1 Shared Gating Network

**Before (v1/v2 - WRONG):**
```python
# Each layer had its own gate
self.activations = [
    SpatiallyVaryingActivation(n_experts=8)  # Creates new gate
    for _ in range(n_layers)
]
```

**After (v3 - CORRECT):**
```python
# One shared gate for all layers
self.shared_gate = SharedGatingNetwork(n_experts=8)
self.activations = [
    SpatiallyVaryingActivation(self.shared_gate, n_experts=8)
    for _ in range(n_layers)
]
```

### 5.2 SIREN Configuration

**Before (v1/v2 - WRONG):**
```python
# Used omega_0=30 for ALL layers
class SineActivation:
    def __init__(self, omega_0=30.0):  # Wrong for subsequent layers
```

**After (v3 - CORRECT):**
```python
# Matches SatCLIP: w0_initial=30 for first, w0=1 for rest
for i in range(n_layers):
    is_first = (i == 0)
    layer_w0 = 30.0 if is_first else 1.0
    layers.append(SirenLayer(dim_in, dim_out, w0=layer_w0, is_first=is_first))
```

### 5.3 HybridEncoder Initialization

**Before (v2 - WRONG):**
```python
# Used Kaiming init even for SIREN
def _init_weights(self):
    for linear in self.linears:
        nn.init.kaiming_normal_(linear.weight)  # Wrong for SIREN
```

**After (v3 - CORRECT):**
```python
if activation == 'siren':
    # Use proper SIREN initialization
    layers.append(SirenLayer(..., is_first=is_first))
else:
    # Kaiming for non-SIREN
    nn.init.kaiming_normal_(linear.weight)
```

### 5.4 Spatial Blocking

**Before (v1/v2 - WRONG):**
```python
# Random split causes spatial leakage
indices = np.random.permutation(n)
train_idx, test_idx = indices[:split], indices[split:]
```

**After (v3 - CORRECT):**
```python
# Grid-based blocking prevents leakage
for point in points:
    cell_id = get_grid_cell(point.lon, point.lat, grid_size=5.0)
    if cell_id in test_cells:
        test_points.append(point)
    else:
        train_points.append(point)
```

### 5.5 Fair SatCLIP Baseline

**Before (v1 - WRONG):**
```python
# MLP head overfits on frozen embeddings
class SatCLIPPredictor:
    def __init__(self):
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )  # Overfits! Gets negative R²
```

**After (v3 - CORRECT):**
```python
# Ridge regression for fair comparison
def evaluate_sklearn(X_train, y_train, X_test, y_test):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return r2_score(y_test, model.predict(X_test))
```

---

## 6. Expected Results

With these fixes, we expect:
1. **SatCLIP baselines** should show positive R² (0.5-0.8 range)
2. **Spatial blocking** will lower all R² scores vs random splits (but more realistic)
3. **Learned activations** should be comparable to or better than ReLU
4. **Spatial activations** may show regional advantages due to shared gating

---

## 7. Files

- `12_learned_activations_v3.ipynb`: Main experiment notebook with all fixes
- `learned_activations_v3_results.csv`: Results table
- `learned_activations_v3_summary.json`: Full summary with configuration
- `learned_activations_v3_comparison.png`: Visualization

---

## 8. Running the Experiments

1. Open `12_learned_activations_v3.ipynb` in Google Colab
2. Run cells 1-3 for setup (clone repo, mount drive, load data)
3. Run cell 4 for spatial blocking data loading
4. Run cells 5-7 for model definitions
5. Run cells 8-9 for evaluation functions
6. Run cell 10 for main experiments
7. Run cell 11 for hybrid experiments
8. Run cells 12-15 for results and visualization
