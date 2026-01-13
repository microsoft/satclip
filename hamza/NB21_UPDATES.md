# NB21/21b/21c: Practical Updates

**Date**: 2026-01-12

---

## Update 1: Reducing to 5 Seeds (Approved)

### Statistical Justification

**Impact of 10 → 5 seeds**:
```
Standard error:   increases by ~41% (sqrt(2) ratio)
95% CI width:     increases by ~23% (t-critical changes)
Statistical power: Still good if CV < 20%
```

**Practical benefits**:
```
Compute savings:
NB21:  10 hours → 5 hours  (50% faster)
NB21b: 12 hours → 6 hours  (50% faster)
NB21c:  7 hours → 3.5 hours (50% faster)

Total: ~12 hours saved → faster iteration
```

### Decision Rule

**5 seeds is sufficient when**:
- ✅ Low variance (CV < 20%): CIs still tight
- ✅ Effect size > 1%: Enough power to detect
- ✅ Consistent direction: Clear winner

**Add more seeds if**:
- ⚠️ Moderate variance (CV 20-50%): Consider +5 more
- ❌ High variance (CV > 50%): Even 10 seeds won't help

### Implementation

**For NB21c**: Already ready to edit - just change:
```python
SEEDS = range(42, 52)  # 10 seeds
```
to:
```python
SEEDS = range(42, 47)  # 5 seeds (compute efficiency)
```

**For NB21/21b**: If not yet started, reduce to 5 seeds. If running, keep 10 (already invested compute).

---

## Update 2: L=10 vs L=40 for Regional Tasks

### Key Finding

> "With smaller coverage size, L=40 was able to be trained to be better than L=10 in some cases, mainly if the region being tested is smaller."

This makes sense: **SH level should match spatial scale**.

**Why**:
- L=40 (1681 dims) captures higher frequencies than L=10 (121 dims)
- Smaller regions → higher relative frequency content
- Global: Low frequencies dominate → L=10 sufficient
- Regional: High frequencies matter → L=40 helps

### Where to Test

#### Option A: NB21 Modification (Best fit)
If NB21 hasn't started regional experiments yet:

```python
# Add to NB21 Exp 3:
REGIONS = ['himalayas', 'sahara']
SAMPLE_SIZES = [10000, 20000]
SH_LEVELS = [10, 40]  # NEW: Test both
ACTIVATIONS = ['relu', 'spline']
SEEDS = 5  # Reduced
```

This adds ~5 hours compute (2 regions × 2 sizes × 2 L × 5 seeds × 2 acts × 100s).

#### Option B: Quick Follow-Up Experiment
If NB21 already running, create small standalone experiment:

```python
# L=10 vs L=40 Regional Comparison
REGIONS = ['himalayas', 'sahara']
N_SAMPLES = 20000  # Larger sample only
SH_LEVELS = [10, 40]
ACTIVATIONS = ['relu', 'spline']
SEEDS = 5

Expected outcome:
- Himalayas (complex terrain): L=40 > L=10
- Sahara (flat): L=10 ≈ L=40
```

Compute: ~2.5 hours (2 regions × 2 L × 5 seeds × 2 acts × 100s).

#### Option C: Post-Hoc Analysis
If NB21 already has L=10 regional results:
1. Run quick L=40 experiment (just 20K samples)
2. Compare to NB21's L=10 results
3. Document difference in regional performance

### Hypothesis

**Expected pattern**:
```
Global task (large coverage):
  L=10:  R² = 0.90
  L=40:  R² = 0.90  (no improvement, just more params)

Regional task (small coverage):
  L=10:  R² = 0.75  (underfit - missing high frequencies)
  L=40:  R² = 0.82  (better - captures detail)
```

### Integration

After running L=40 regional experiment:

**Add to synthesis**:
```python
# Final recommendation table:
Scale      L-value    ReLU/Spline    Why
Global     L=10       ReLU           Low-freq dominant
Regional   L=40       ReLU/Spline    High-freq matters
Local      L=40+      Spline?        Very high-freq
```

---

## Update 3: NB21c Note (Raw Coords)

**L=10 vs L=40 doesn't apply to NB21c** because:
- NB21c uses raw coordinates (no SH)
- L-value is SH-specific parameter
- Raw coords test different hypothesis

**But NB21c can inform L choice**:
- If Raw+acts >> SH(L=10)+acts → use higher L
- If SH(L=10) sufficient → stick with L=10

---

## Recommended Execution Order

### Immediate
1. **If NB21/21b not started**: Reduce to 5 seeds
2. **If NB21/21b running**: Keep 10 seeds (sunk cost)
3. **Launch NB21c with 5 seeds** (~3.5 hours)

### After NB21/21b Complete
1. Analyze results (Priority 2: Performance/param analysis)
2. Cross-task synthesis (Priority 5)
3. **If NB21 didn't test L=40**: Run quick L=40 regional experiment (~2.5 hours)

### After NB21c Complete
1. Raw vs SH comparison
2. Decide on next steps (coastline task, etc.)

---

## Summary

**5 seeds**: ✅ Approved - saves ~12 hours, sufficient if CV < 20%

**L=40 regional**: ✅ Good idea - test SH level scaling
- Add to NB21 if possible
- Otherwise run as quick follow-up
- ~2.5-5 hours depending on scope

**Total new compute**: ~3.5 hours (NB21c) + ~2.5 hours (L=40) = **6 hours**

vs original plan: ~29 hours (all experiments with 10 seeds)

**Savings**: ~23 hours → **79% faster iteration!**
