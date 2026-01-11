# Analysis: Notebook 15 Results

## Summary of Results

### Performance Comparison (R² scores)

| Model | 15-min | 30-min | 1-degree | Params | Efficiency (avg) |
|-------|--------|--------|----------|--------|------------------|
| **RFF (n=25)** | **0.733** | 0.771 | **0.794** | 231K | **0.033** |
| **Spline (k=10)** | 0.728 | **0.777** | 0.783 | 231K | **0.033** |
| **ReLU baseline** | 0.722 | 0.761 | 0.787 | 231K | 0.032 |
| SatCLIP L=10 | 0.704 | 0.744 | 0.767 | 446K | 0.017 |
| SatCLIP L=40 | 0.618 | 0.656 | 0.673 | 1.2M | 0.005 |

### Key Findings

#### 1. Learned Activations Outperform SatCLIP
- **RFF/Spline beat SatCLIP L=10 by 2.7-3.3%** across all resolutions
- **RFF/Spline beat SatCLIP L=40 by 11-12%** (!!!)
- Even basic ReLU beats SatCLIP L=40

#### 2. SatCLIP L=40 Severely Underperforms
- L=40 performs **8-12% worse than L=10** despite 16× more features
- L=40 has 1600 SH features vs L=10's 100 features
- Suggests Ridge regression struggles with high-dimensional SH space
- Possible overfitting or feature redundancy

#### 3. Efficiency Advantage is Massive
- RFF/Spline: **2× better** efficiency than SatCLIP L=10
- RFF/Spline: **6× better** efficiency than SatCLIP L=40
- Our models use ~50% fewer parameters than L=10

#### 4. RFF vs Spline: Photo Finish
- RFF wins at 15-min and 1-degree
- Spline wins at 30-min
- Differences are small (0.5-1%), likely within noise

#### 5. Resolution Scaling
All models improve at coarser resolutions (easier task):
- 15-min → 1-degree: RFF improves from 0.733 → 0.794 (+6.1%)
- Learned activations maintain advantage across all scales

---

## Critical Questions

### Q1: Why is SatCLIP L=40 so bad?

**Hypotheses:**
1. **Ridge alpha not tuned**: Default α=1.0 might be wrong for 1600-dim
2. **Feature redundancy**: High-order SH features are correlated
3. **Curse of dimensionality**: Ridge can't handle 1600 features with 10K samples
4. **Frozen embeddings**: L=40 needs fine-tuning, not frozen evaluation

**Test:** Try different Ridge alphas (0.1, 1.0, 10.0, 100.0) for L=40

### Q2: Should we fine-tune SatCLIP end-to-end?

**Current setup:** Freeze SatCLIP encoder, train Ridge head
**Alternative:** Unfreeze encoder, train full model end-to-end

**But:** ARCHITECTURE_SETUP.md said MLP heads overfit on frozen embeddings, which is why we use Ridge. If we fine-tune, we're changing what we're evaluating (not just the encoder).

### Q3: Is 100 epochs enough?

Learned activations have more parameters to optimize (coefficients, knots). Maybe need more iterations?

**Test:** Train for 200 epochs, compare convergence

### Q4: Are we comparing fairly?

Our models: 231K params
SatCLIP L=10: 446K params
SatCLIP L=40: 1.2M params

Should we:
- Match parameter counts by adding layers/width to our models?
- Report parameter-normalized metrics (already doing with efficiency)?

---

## Experimental Setup Issues

### Issues We Should Fix:

1. **No multiple runs**: Only 1 run per config
   - **Fix**: Run 3-5 times with different seeds, report mean ± std

2. **No hyperparameter tuning**: Same LR, epochs, batch size for all
   - **Fix**: Grid search LR, architecture for each model type

3. **Single train/test split**: Same spatial blocking seed
   - **Fix**: Test on multiple spatial blocking seeds

4. **No early stopping**: Training for fixed 100 epochs
   - **Fix**: Add validation set, early stop on validation R²

5. **No learning rate schedule**: Constant 1e-3
   - **Fix**: Try cosine annealing or ReduceLROnPlateau

### Issues That Are Actually Fine:

1. **Parameter mismatch**: Efficiency metric accounts for this
2. **Ridge for SatCLIP**: This is the recommended fair evaluation
3. **Spatial blocking**: Prevents leakage, correct approach

---

## Next Steps (Prioritized)

### Immediate (Notebook 16):
**Phase 1 Core Comparison** - The 2×2 grid from EXPERIMENT_ROADMAP.md

Test at 15-min resolution only:
1. Raw + SIREN
2. Raw + RFF (n=25, 50, 100)
3. Raw + Spline (k=10, 20, 30)
4. SH(L=10) + SIREN ← **SatCLIP baseline**
5. SH(L=10) + RFF (n=25, 50, 100)
6. SH(L=10) + Spline (k=10, 20, 30)

**Win condition**: Does SH + RFF beat SH + SIREN?

### Short-term (Notebook 17):
**Investigate SatCLIP L=40 failure**

Test different Ridge alphas for SatCLIP L=40:
- α = 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0
- Plot R² vs α to find optimal regularization

### Medium-term (Notebook 18):
**RFF/Spline variant ablation**

RFF variants:
- n_features: 25, 50, 100, 200
- learnable_freq: True/False
- max_freq: 5, 10, 20, 50
- freq_init: 'linear', 'log', 'random'

Spline variants:
- n_knots: 10, 20, 30, 50
- init: 'relu', 'linear', 'zero', 'random'
- input_range: (-3, 3), (-5, 5), (-10, 10)
- learnable knot positions

### Long-term:
1. Add spatial gating (MoE) - Notebook 19
2. Test on MOSAIKS data - Notebook 20
3. Scale to contrastive training - Notebook 21

---

## Implications for Learned Activations Hypothesis

### What We've Proven:
✅ Learned activations (RFF/Spline) work without SH features
✅ They beat SatCLIP L=10 in both R² and efficiency
✅ They scale well across resolutions
✅ They're easy to train (100 epochs, no special tricks)

### What We Still Need to Test:
❓ Do learned acts + SH beat SIREN + SH? (Phase 1 core question)
❓ Which RFF/Spline variant is best?
❓ Do learned acts help with spatial variation? (Phase 4)
❓ Do they scale to contrastive training? (Phase 5)

### Why Dan's Hypothesis Looks Good:
Our simple RFF/Spline already beat the SatCLIP baseline. If we can **combine them with SH features**, we might get even better performance. This is exactly what Phase 1 of the roadmap is testing!

**Prediction**: SH(L=10) + RFF will beat SH(L=10) + SIREN because:
- SH provides the frequencies
- RFF provides a more expressive nonlinearity than sine
- Joint optimization learns better feature combinations

---

## Open Questions

1. **Why did RFF win in notebook 14 but Spline won at 30-min?**
   - Random initialization differences?
   - Different optimal frequency/knot spacing for different resolutions?
   - Need multiple runs to see if this is real or noise

2. **What's the right number of RFF features?**
   - 25 seems to work well
   - Should we try 50 or 100? (Dan suggested this)
   - Trade-off between expressiveness and overfitting

3. **Can we visualize what the learned activations are doing?**
   - Plot learned activation shapes after training
   - Compare to ReLU and SIREN
   - Do they look different at different layers?

4. **Should we try other activation families?**
   - Rational functions (Padé approximants)
   - Wavelets
   - Polynomials (Chebyshev, Legendre)
