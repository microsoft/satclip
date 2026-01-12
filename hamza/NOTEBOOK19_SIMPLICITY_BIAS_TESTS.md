# Notebook 19: Testing the Simplicity Bias Hypothesis

**Date**: 2026-01-12
**Status**: Planning
**Goal**: Find "alpha" - identify tasks/setups where learned activations excel over ReLU + SH

---

## Motivation: Where's the Advantage?

Based on Teney et al. (2024) "Do We Always Need the Simplicity Bias?", we know:

### ✅ Where Learned Activations Excel (Paper Findings):
1. **Regression tasks** (+large improvement) vs classification
2. **High-frequency tasks** (elevation, sharp transitions)
3. **Complex functions** (higher Total Variation correlates with better performance)
4. **Tabular data** (axis-aligned boundaries)

### ❌ Where ReLU is Near-Optimal (Paper Findings):
1. **Image classification** (learned acts → GeLU-like, minimal gain)
2. **Smooth/low-frequency tasks** (simplicity bias matches task)

### 🤔 Our Results So Far (NB18):
- **Population density** (smooth, low-frequency): ReLU wins by 0.63%
- **Spline beats SIREN** by +1.88%
- **Conclusion**: Population density is too smooth to benefit from learned activations

### 🎯 Hypothesis for Notebook 19:
**We need to test on tasks/setups where the simplicity bias is DETRIMENTAL:**
1. High-frequency geographic tasks (elevation, coastlines)
2. Regression formulation (vs classification)
3. Higher spatial resolution (finer grids → more high-frequency content)
4. Tasks with sharp transitions (urban boundaries, terrain features)

---

## Experiment Design

### Experiment 1: Regression vs Classification Formulation ⭐⭐⭐ (CRITICAL)

**Paper Finding**: "Regression is more difficult for NNs than classification, and the simplicity bias of ReLUs is partly to blame. Learned activations improve performance by helping networks represent more complex functions."

**Setup**:
```python
# Same population density data, different loss functions

# Classification (current approach)
task_cls = {
    'target': 'log_density_binned',  # 100 bins
    'loss': 'CrossEntropyLoss',
    'metric': 'accuracy',
    'output_activation': 'softmax'
}

# Regression (direct prediction)
task_reg = {
    'target': 'log_density_continuous',  # raw log density
    'loss': 'MSELoss',
    'metric': 'R² (or MSE)',
    'output_activation': 'linear'
}

# Test on BOTH formulations
activations = ['relu', 'spline_k15', 'siren']
input_encoding = 'sh_L10'  # SH(L=10) features

# Train 3 models × 2 tasks = 6 models
```

**Expected Outcome**:
- Classification: ReLU ≈ Spline (as we've seen)
- Regression: **Spline > ReLU** (paper predicts this)

**Analysis**:
1. Compare R² (regression) vs accuracy (classification) improvement
2. Measure function complexity (Total Variation - see Experiment 4)
3. Visualize prediction surfaces for regression vs classification
4. Plot accuracy vs complexity (expect correlation only for regression)

**Why This Matters**:
- If Spline beats ReLU on regression but not classification → confirms paper's hypothesis
- Regression is closer to real-world use case (predicting continuous values)
- Could be a major finding: "use learned activations for regression tasks"

---

### Experiment 2: High-Frequency Geographic Tasks ⭐⭐⭐ (CRITICAL)

**Paper Finding**: "For regression tasks and tabular data, new learned activations greatly improve accuracy by helping learn complex functions."

**Setup**:
```python
# 3 tasks of increasing frequency content

# Task A: Population Density (BASELINE - smooth, low-frequency)
task_pop = {
    'data': 'GPW population density',
    'characteristics': 'Smooth, low-frequency, gradual transitions',
    'expected': 'ReLU ≈ Spline (as seen in NB18)'
}

# Task B: Elevation (HIGH-FREQUENCY)
task_elev = {
    'data': 'ETOPO1 elevation (meters)',
    'characteristics': 'Sharp peaks, valleys, mountains, discontinuities',
    'expected': '**Spline > ReLU** (paper predicts)'
}

# Task C: Temperature Gradient Magnitude (MEDIUM-FREQUENCY)
task_temp = {
    'data': '|∇T| from climate reanalysis',
    'characteristics': 'Sharp fronts, medium-frequency',
    'expected': 'Spline ≥ ReLU'
}

# Task D: Coastline Distance (SHARP TRANSITIONS)
task_coast = {
    'data': 'Distance to nearest coastline',
    'characteristics': 'Step function at coast, sharp boundaries',
    'expected': '**Spline >> ReLU** (extreme test)'
}

# For each task:
activations = ['relu', 'spline_k15', 'siren']
input_encoding = 'sh_L10'
formulation = 'regression'  # Use regression based on Exp 1 findings
```

**Expected Outcome**:
- Population: ReLU wins (as seen)
- Elevation: **Spline wins** (high-frequency)
- Coastline: **Spline wins significantly** (sharp transitions)
- Temperature: Spline ≥ ReLU (medium)

**Analysis**:
1. Rank tasks by frequency content (use Fourier analysis or visual inspection)
2. Plot: Task frequency → Spline advantage
3. Measure prediction error at high-frequency regions (e.g., mountain peaks)
4. Visualize learned activation shapes per task

**Why This Matters**:
- Directly tests paper's hypothesis
- If Spline beats ReLU on elevation → major finding
- Identifies domain-specific advantages

**Data Sources**:
- ETOPO1: Global elevation (1 arc-minute resolution)
- ERA5: Temperature reanalysis
- GSHHG: Coastline database

---

### Experiment 3: Multi-Resolution Analysis ⭐⭐ (HIGH PRIORITY)

**User Motivation**: "could be really good for smaller resolution" (finer spatial grids)

**Hypothesis**: Finer resolution → more high-frequency detail → learned activations shine

**Setup**:
```python
# Use elevation data (from Exp 2) at multiple resolutions

resolutions = [
    {'name': 'Coarse',  'spacing': '1 degree',    'samples': ~10K},
    {'name': 'Medium',  'spacing': '0.5 degree',  'samples': ~40K},
    {'name': 'Fine',    'spacing': '0.25 degree', 'samples': ~160K},
    {'name': 'Ultra',   'spacing': '0.1 degree',  'samples': ~1M},  # if feasible
]

# For each resolution:
activations = ['relu', 'spline_k15', 'siren']
input_encoding = 'sh_L10'
task = 'elevation_regression'

# Measure:
#   - R² performance
#   - Spline advantage (Spline R² - ReLU R²)
#   - Function complexity (TV)
```

**Expected Outcome**:
- Coarse: ReLU ≈ Spline (low frequency content, smooth)
- Fine: **Spline > ReLU** (high frequency content, details)
- Ultra: **Spline >> ReLU** (very high frequency)

**Analysis**:
1. Plot: Resolution → Spline advantage
2. Measure frequency content at each resolution (via Fourier spectrum)
3. Identify crossover point (where Spline starts winning)
4. Check if finer resolution correlates with higher complexity

**Why This Matters**:
- Tests user's intuition about "smaller resolution"
- Practical guidance: use learned activations for high-res tasks
- Could be major finding: "resolution-dependent activation choice"

---

### Experiment 4: Function Complexity Measurement ⭐⭐

**Paper Method**: Total Variation (TV) as complexity measure

**Formula** (from paper):
```
TV(f, T) = E[x1,x2 ~ T] ∫[x1 to x2] |f'(x)| dx

Approximation:
TV(f, T) ≈ E[xa,xz ~ T] Σ |f(xi+1) - f(xi)|
```

**Setup**:
```python
# For each trained model, measure complexity

def total_variation(model, test_data, n_paths=1000):
    """
    Measure TV along random paths in input space.

    Args:
        model: Trained neural network
        test_data: Test dataset for sampling endpoints
        n_paths: Number of random paths to average over

    Returns:
        tv: Total variation (complexity measure)
    """
    tv_values = []

    for _ in range(n_paths):
        # Sample two points from test set
        x1, x2 = random.sample(test_data, 2)

        # Create path: x(λ) = (1-λ)x1 + λx2, λ ∈ [0,1]
        lambdas = np.linspace(0, 1, 100)
        path = [(1-lam)*x1 + lam*x2 for lam in lambdas]

        # Evaluate model along path
        outputs = [model(x) for x in path]

        # Compute TV: sum of |f(xi+1) - f(xi)|
        tv = sum(abs(outputs[i+1] - outputs[i]) for i in range(len(outputs)-1))
        tv_values.append(tv)

    return np.mean(tv_values)

# Measure TV for all models
models = {
    'ReLU + SH': model_relu,
    'Spline + SH': model_spline,
    'SIREN + SH': model_siren,
}

complexities = {name: total_variation(model, test_data) for name, model in models.items()}
```

**Analysis**:
1. Plot: Complexity (TV) vs R² for different activations
2. Check if **complexity correlates with performance** (paper finding for regression)
3. Compare TV across tasks (population vs elevation vs coastline)
4. Visualize function along random paths (smooth vs jagged)

**Expected Outcome** (from paper):
- **Regression**: Accuracy ↑ as complexity ↑ (positive correlation)
- **Classification**: No clear correlation
- **Spline models**: Higher TV than ReLU models
- **High-frequency tasks**: Require higher TV for good performance

**Why This Matters**:
- Quantifies "simplicity bias" precisely
- Validates paper's hypothesis on our data
- Provides diagnostic tool for future tasks

---

### Experiment 5: Task Difficulty Scaling ⭐

**Hypothesis**: Harder tasks → more need for expressive activations

**Setup**:
```python
# Create tasks of varying difficulty from same base data

# Use elevation data, vary difficulty by:

difficulty_configs = [
    {
        'name': 'Easy',
        'target': 'elevation_smoothed_50km',  # Heavily smoothed
        'expected_baseline_R2': 0.95,
    },
    {
        'name': 'Medium',
        'target': 'elevation_smoothed_10km',  # Moderately smoothed
        'expected_baseline_R2': 0.80,
    },
    {
        'name': 'Hard',
        'target': 'elevation_raw',  # Full complexity
        'expected_baseline_R2': 0.65,
    },
    {
        'name': 'Very Hard',
        'target': 'elevation_gradient_magnitude',  # |∇elevation|
        'expected_baseline_R2': 0.40,
    },
]

# For each difficulty:
activations = ['relu', 'spline_k15']
```

**Expected Outcome**:
- Easy: ReLU ≈ Spline (simple function, simplicity bias helps)
- Medium: ReLU ≈ Spline (moderate complexity)
- Hard: **Spline > ReLU** (complex function, need expressiveness)
- Very Hard: **Spline >> ReLU** (very complex, simplicity bias detrimental)

**Analysis**:
1. Plot: Task difficulty → Spline advantage
2. Measure complexity (TV) at each difficulty level
3. Identify difficulty threshold where Spline starts winning
4. Check if gradient magnitude task is too hard for both

**Why This Matters**:
- Practical guidance: when to use learned activations
- Tests hypothesis: harder tasks need expressive activations
- Could inform active learning / curriculum design

---

## Summary: What We're Testing

| Experiment | Hypothesis | Expected Result | Priority |
|------------|------------|-----------------|----------|
| **1. Regression vs Classification** | Learned acts help regression more | **Spline > ReLU** for regression | ⭐⭐⭐ CRITICAL |
| **2. High-Frequency Tasks** | Sharp transitions favor learned acts | **Spline > ReLU** on elevation/coastline | ⭐⭐⭐ CRITICAL |
| **3. Multi-Resolution** | Finer resolution → more advantage | **Spline > ReLU** at fine resolution | ⭐⭐ HIGH |
| **4. Complexity Measurement** | TV correlates with performance | Positive correlation for regression | ⭐⭐ HIGH |
| **5. Task Difficulty** | Harder tasks need expressiveness | **Spline > ReLU** on hard tasks | ⭐ MEDIUM |

---

## Success Criteria

Notebook 19 is successful if we can answer:

### 🎯 Primary Question: "When do learned activations beat ReLU + SH?"

**Expected Answers**:
1. ✅ **Regression formulation** (not classification)
2. ✅ **High-frequency tasks** (elevation, coastlines, gradients)
3. ✅ **Fine spatial resolution** (finer grids → more detail)
4. ✅ **Hard tasks** requiring complex functions

### 📊 Secondary Questions:

1. **Does complexity correlate with performance?**
   - Yes for regression (paper predicts)
   - No for classification

2. **What's the magnitude of improvement?**
   - Small on smooth tasks (<1%)
   - Large on high-frequency tasks (>5%?)

3. **Is there a resolution threshold?**
   - Below X km → ReLU wins (coarse, smooth)
   - Above X km → Spline wins (fine, detailed)

---

## Deliverables

After Notebook 19, we will have:

1. **Actionable Guidelines**:
   - Use ReLU for: classification, smooth/low-frequency tasks, coarse resolution
   - Use Spline for: regression, high-frequency tasks, fine resolution

2. **Quantitative Evidence**:
   - R² improvements on elevation vs population
   - Complexity (TV) measurements correlating with performance
   - Resolution threshold for Spline advantage

3. **Publication-Ready Results**:
   - "Learned activations excel on high-frequency geographic regression tasks"
   - "Simplicity bias is detrimental for elevation prediction"
   - "Finer resolution benefits from learned activations"

4. **Next Steps Identified**:
   - If Spline wins on elevation → test on other high-freq tasks
   - If regression helps → explore other regression problems
   - If resolution matters → optimize for specific resolution ranges

---

## Updated Notebook Priorities

### New Order (Phase 2):

| Notebook | Focus | Priority | Status |
|----------|-------|----------|--------|
| **18** | Spline deep dive | ⭐⭐⭐ | ✅ Complete |
| **19** | **Simplicity bias tests** | ⭐⭐⭐ CRITICAL | 📝 This document |
| **20** | Complexity analysis + visualization | ⭐⭐ | Pending |
| **21** | Robustness (multiple seeds) | ⭐⭐ | Pending |
| **22** | Architecture interaction | ⭐ | Pending (was old NB19) |
| **23** | Raw + Learned deep dive | ⭐ | Pending |

### Rationale for New Priority:
- **NB19 (new)**: Tests fundamental hypothesis from paper - MUST DO FIRST
- Old NB19 (architecture) → moved to NB22 (less critical than finding alpha)
- NB20 (visualization) → important for understanding NB19 results
- NB21 (robustness) → validate findings from NB19

---

## Estimated Timeline

**Notebook 19 Compute Time**:
- Exp 1 (Regression vs Classification): ~30 min (6 models)
- Exp 2 (High-Frequency Tasks): ~2 hours (12 models × 4 tasks)
- Exp 3 (Multi-Resolution): ~1.5 hours (12 models × 3-4 resolutions)
- Exp 4 (Complexity Measurement): ~30 min (analysis only, no training)
- Exp 5 (Task Difficulty): ~1 hour (8 models × 4 difficulties)

**Total**: ~5-6 hours GPU time (Colab T4)

**Data Preparation**:
- ETOPO1 elevation: Already available
- Temperature gradient: Can compute from ERA5 (or skip if time-limited)
- Coastline distance: Can compute from GSHHG

**Critical Path**:
1. Exp 1 + 2 (MUST DO): ~2.5 hours
2. Exp 3 + 4 (HIGH VALUE): ~2 hours
3. Exp 5 (NICE TO HAVE): ~1 hour

---

## Connection to Paper

This notebook directly tests the core claims from Teney et al. (2024):

| Paper Claim | Our Test |
|-------------|----------|
| "Regression benefits greatly from learned activations" | Exp 1: Regression vs Classification |
| "High-frequency tasks require complex functions" | Exp 2: Elevation, coastlines |
| "Complexity correlates with regression performance" | Exp 4: TV measurement |
| "Image classification: ReLU is near-optimal" | (Already shown: smooth population) |

**Key Difference**:
- **Paper**: Meta-learned activations via bi-level optimization
- **Us**: Hand-designed splines with learned knot values
- **But**: Same underlying principle - overcome simplicity bias

**Advantage of Our Approach**:
- Simpler (no meta-learning)
- Faster (standard backprop)
- More interpretable (visualize knot values directly)
- Already shown to work (NB18)

---

## Open Questions

After Notebook 19, we'll need to address:

1. **Why does Spline work?** (approximation theory, frequency analysis)
2. **Can we predict when to use Spline?** (task characteristics → activation choice)
3. **What about other learned activation families?** (Fourier features, wavelets, KANs)
4. **How do results scale?** (larger models, more data, different domains)

But first, let's find the "alpha" - prove that learned activations have real advantages in specific settings.

---

## Conclusion

**This notebook shifts focus from "comprehensive characterization" to "finding alpha".**

Key philosophy:
- 🎯 Test specific hypotheses from paper
- 🔍 Focus on where learned activations should excel
- 📊 Measure complexity quantitatively
- ✅ Provide actionable guidelines

If Notebook 19 succeeds, we'll have:
- Clear evidence of when learned activations beat ReLU
- Quantitative thresholds (resolution, frequency, difficulty)
- Validated hypotheses from literature
- Foundation for Phase 3 work

**Next step**: Execute Notebook 19 experiments and analyze results.
