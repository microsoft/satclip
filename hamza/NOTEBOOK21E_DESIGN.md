# Notebook 21e: Fast Follow-Up Experiments

**Created**: 2026-01-13
**Purpose**: Fill critical gaps from NB21/21b/21c with streamlined experiments
**Runtime**: ~3 hours total

---

## Context: What We Already Know

### Major Discovery from NB21/21b/21c ⭐

**SH encoding masks learned activation benefits!**

| Task | With SH (NB21/21b) | With Raw Coords (NB21c) | SH Effect |
|------|-------------------|------------------------|-----------|
| **Elevation** | ReLU wins (-0.11%) | **Spline wins (+6.09% ± 2.10%)** | SH masked **+6.2%** gain |
| **Population** | ReLU wins (-0.23%) | **Spline wins (+8.51% ± 5.60%)** | SH masked **+8.7%** gain |

**This confirms all 4 experts' hypothesis!**

---

## What We Still Need to Test

### 1. **Does L=40 help for regional tasks?** (from NB21d)
- Your observation: L=40 outperforms L=10 for smaller regions
- Test: Himalayas & Sahara with L=10 vs L=40
- Question: Is the parameter cost worth it?

### 2. **Does RFF work with raw coords?** (mentioned in original plan)
- RFF = Random Fourier Features (alternative to Spline)
- Test: Raw+RFF vs Raw+Spline vs Raw+ReLU
- Question: Is RFF competitive?

### 3. **Verify population result** (high variance in NB21c)
- NB21c: +8.51% ± 5.60% (CV ~66% - high!)
- Add 3 more seeds to reduce uncertainty
- Question: Is this effect real or noise?

---

## Notebook Structure

### Experiment 1: L=10 vs L=40 Regional (~1.5 hours)
**Config**: 2 regions × 2 L-values × 2 acts × 3 seeds = 24 runs

**Regions**:
- Himalayas (70-100°E, 25-40°N) - Complex terrain
- Sahara (-10-40°E, 15-35°N) - Flat terrain

**L-values**:
- L=10: 121 dims (baseline)
- L=40: 1,681 dims (13× more parameters)

**Activations**: ReLU, Spline

**Analysis**:
- Compare R² improvement vs parameter cost
- Statistical significance (paired t-test)
- Decision rule: Worth it if improvement >1% AND p<0.05

**Expected outcomes**:
- If L=40 helps: Regional tasks need higher frequencies
- If L=10 sufficient: Current setup is optimal
- If terrain-dependent: Guidance for practitioners

---

### Experiment 2: Raw + RFF (~1 hour)
**Config**: 2 tasks × 3 configs × 3 seeds = 18 runs

**Tasks**:
- Elevation (high-frequency)
- Population (smooth)

**Configs**:
- Raw+ReLU (baseline)
- Raw+Spline (from NB21c)
- Raw+RFF (new test)

**Analysis**:
- Is RFF competitive with Spline?
- Task-dependent patterns?
- Training efficiency

**Expected outcomes**:
- If RFF ≈ Spline: Simpler alternative (no learnable knots)
- If Spline > RFF: Confirms splines are special
- If task-dependent: Guidance on when to use what

---

### Experiment 3: Population Verification (~30 min)
**Config**: 2 acts × 3 new seeds = 6 runs

**Setup**:
- Use seeds 45, 46, 47 (different from NB21c's 42-51)
- Same task: Population density, 15K samples
- Same encoding: Raw coordinates

**Analysis**:
- Combine with NB21c results (13 seeds total)
- Reduced variance estimate
- Confirm if +8.51% is real

**Expected outcomes**:
- If consistent: Population effect confirmed
- If inconsistent: High variance is fundamental issue
- Either way: Better uncertainty estimate

---

## Key Design Decisions

### Why 3 seeds instead of 5-10?
- **Speed**: 3 seeds = 50-70% faster than 10 seeds
- **Still valid**: Sufficient if CV < 20%
- **Iterative**: Can add more if variance high

### Why these specific experiments?
- **Exp 1**: User's direct observation (L=40 works better regionally)
- **Exp 2**: Missing from original plan, RFF never tested
- **Exp 3**: High-variance result needs verification

### What about other missing pieces?
- **Performance/parameter analysis**: Post-processing, no new runs needed
- **Cross-task synthesis**: Combine all notebooks' results
- **Validation split**: Infrastructure change, not critical for conclusions
- **Coastline task**: Only if splines show strong promise (they did!)

---

## Expected Runtime

**Experiment 1**: ~1.5 hours
- 24 runs × ~4 min/run = 96 min

**Experiment 2**: ~1 hour
- 18 runs × ~3.5 min/run = 63 min

**Experiment 3**: ~30 min
- 6 runs × ~5 min/run = 30 min

**Total**: ~3 hours on Colab T4

**Efficiency gains**:
- Reduced seeds (3 vs 10): 70% faster
- Reduced epochs where appropriate: 20% faster
- Focused scope: No redundant experiments

---

## What We'll Learn

### Critical Questions Answered

1. **L=10 vs L=40 regional**: Parameter cost-benefit for small regions
2. **Raw+RFF viability**: Alternative to Spline for practitioners
3. **Population result confidence**: Real effect or high variance?

### Impact on Conclusions

**If L=40 helps regionally**:
- Update recommendation: Use L=40 for regions <5° coverage
- Explains user's observation
- Practical guidance for deployment

**If RFF competitive**:
- Simpler alternative to Splines (no learnable knots)
- Easier to implement and tune
- Broadens "learned activation" story

**If population verified**:
- Confirms SH masking effect on TWO tasks
- Stronger evidence for publication
- Generalizable finding

---

## Integration with Other Notebooks

### Cross-Notebook Synthesis (After 21e)

**Complete dataset**:
- NB21: SH+acts on elevation (10 seeds, 5 experiments)
- NB21b: SH+acts on population (10 seeds, 5 experiments)
- NB21c: Raw+acts on both tasks (10 seeds, 5 experiments)
- NB21e: L=40 regional + RFF + population verification (3 seeds, 3 experiments)

**Analysis tasks**:
1. **Encoding comparison**: Raw vs SH(L=10) vs SH(L=40)
2. **Activation comparison**: ReLU vs Spline vs SIREN vs RFF
3. **Task patterns**: Elevation vs Population differences
4. **Scale effects**: Global vs Regional performance
5. **Parameter efficiency**: R² per 1K parameters

**Final deliverable**: Decision tree for practitioners
```
Task: Geographic prediction
├─ Coverage: Global (>1000km²)
│  ├─ Use SH(L=10) + ReLU
│  └─ Reason: SH pre-smooths, learned acts don't help
│
└─ Coverage: Regional (<100km²)
   ├─ High-frequency terrain (mountains, coastlines)
   │  ├─ Use Raw + Spline OR SH(L=40) + ReLU
   │  └─ Reason: Capture local detail
   │
   └─ Smooth terrain (plains, ocean)
      ├─ Use Raw + ReLU OR SH(L=10) + ReLU
      └─ Reason: Simpler is sufficient
```

---

## Success Criteria

### NB21e succeeds if:

1. ✅ **L=40 analysis complete**: Clear cost-benefit for regional tasks
2. ✅ **RFF tested**: Know if it's competitive with Spline
3. ✅ **Population verified**: Reduced uncertainty on NB21c result
4. ✅ **Fast execution**: Completes in <4 hours
5. ✅ **Clear conclusions**: Each experiment answers its question

**Even negative results are valuable:**
- If L=40 doesn't help: L=10 is optimal, save parameters
- If RFF fails: Splines are special, worth the complexity
- If population inconsistent: High variance is fundamental, need different approach

---

## After NB21e: Complete Analysis Pipeline

1. **Run NB21e** (~3 hours)
2. **Performance/parameter analysis** (~1 hour)
   - Load results from all notebooks
   - Calculate R² per 1K parameters
   - Pareto frontier plots
3. **Cross-task synthesis** (~2 hours)
   - Compare all configs systematically
   - Statistical tests across experiments
   - Identify robust patterns
4. **Create decision tree** (~1 hour)
   - Practitioner guidance
   - When to use what
   - Parameter cost-benefit
5. **Write up results** (~4 hours)
   - Either positive result (SH masking) or negative (simplicity sufficient)
   - Both are publishable with this level of rigor

**Total time to publication-ready results**: ~11 hours after NB21e completes

---

## Bottom Line

**NB21e is the last experimental notebook needed before write-up.**

After this completes, we have:
- ✅ Multi-seed validation (NB21/21b/21c)
- ✅ Raw vs SH comparison (NB21c)
- ✅ Regional vs global (NB21/21c)
- ✅ L-value comparison (NB21e Exp 1)
- ✅ RFF validation (NB21e Exp 2)
- ✅ High-variance verification (NB21e Exp 3)

All critical experiments complete. Ready for synthesis and publication.
