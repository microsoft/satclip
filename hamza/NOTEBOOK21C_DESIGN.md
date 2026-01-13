# Notebook 21c: Raw Coordinates Reproducibility Design

**Created**: 2026-01-12
**Purpose**: Test learned activations WITHOUT SH encoding to determine if SH masks their benefits

---

## Critical Question

**Does SH(L=10) pre-encoding mask learned activation benefits?**

All 4 expert reviewers identified this as the **critical missing control experiment**.

---

## Background

### Current Evidence (with SH encoding)

**NB19/19b (single seed)**:
- Elevation: SH+Spline +0.36% over SH+ReLU
- Population: SH+Spline -0.64% (ReLU wins)

**NB21/21b (multi-seed, in progress)**:
- Will establish reproducible baselines with SH(L=10)
- 10 seeds per configuration
- Statistical rigor: mean±std, 95% CI, paired t-tests

### Gap: No Multi-Seed Raw Coordinate Baseline

**NB14-16 tested raw coords** but:
- ❌ Single seed only (can't trust small differences)
- ❌ No statistical validation
- ❌ No direct comparison to SH results

**Expert consensus** (4/4 experts):
> "The simplicity paper uses **raw coordinates**, not SH. Need to test if SH pre-smooths signals and removes content learned activations could capture."

---

## Three Critical Comparisons

### 1. Does SH Add Value?

**Compare**: Raw+ReLU vs SH+ReLU (from NB21/21b)

**Possible outcomes**:
- **SH >> Raw** (e.g., +5% R²): SH encoding is essential, justifies parameter cost
- **SH ≈ Raw** (e.g., ±1% R²): SH is redundant, raw coords sufficient
- **Raw > SH**: Unexpected, would suggest SH hurts

**Example**:
```
Elevation task, 10 seeds:
  Raw+ReLU:  R² = 0.73 ± 0.02
  SH+ReLU:   R² = 0.90 ± 0.01
  → SH provides +23% absolute improvement
  → Conclusion: SH encoding essential
```

---

### 2. Does SH Mask Spline Gains?

**Compare**: (Raw+Spline - Raw+ReLU) vs (SH+Spline - SH+ReLU)

**Hypothesis**: SH pre-smooths signals, leaving no high-frequency content for splines to capture

**Possible outcomes**:

**Outcome A: SH masks gains**
```
Raw coords:   Spline advantage = +3.0% ± 0.5%
SH coords:    Spline advantage = +0.3% ± 0.5%
→ Spline gains 10× larger without SH
→ Conclusion: SH pre-encoding masks spline benefits
→ Recommendation: Use Raw+Spline when SH not needed
```

**Outcome B: SH enhances gains**
```
Raw coords:   Spline advantage = +0.5% ± 0.8%
SH coords:    Spline advantage = +2.0% ± 0.5%
→ Spline gains 4× larger with SH
→ Conclusion: SH and splines are complementary
→ Recommendation: Always use SH+Spline together
```

**Outcome C: No masking**
```
Raw coords:   Spline advantage = +0.3% ± 0.5%
SH coords:    Spline advantage = +0.4% ± 0.5%
→ Similar gains in both cases
→ Conclusion: SH doesn't mask, splines just don't help much
```

---

### 3. Can Splines Win Without SH?

**Compare**: Raw+Spline vs Raw+ReLU (both 10 seeds)

**Hypothesis**: Simplicity paper's effects appear strongest on raw inputs

**Possible outcomes**:

**Outcome A: Splines win decisively**
```
Raw+ReLU:   R² = 0.720 ± 0.015
Raw+Spline: R² = 0.745 ± 0.012
Advantage: +3.5% ± 0.8%, 95% CI: [+2.2%, +4.8%]
CV: 23% (acceptable)
→ Spline wins, confirms simplicity paper
→ Next: Investigate why SH masked this in NB19/21
```

**Outcome B: ReLU wins**
```
Raw+ReLU:   R² = 0.730 ± 0.010
Raw+Spline: R² = 0.725 ± 0.015
Advantage: -0.7% ± 1.2%, 95% CI: [-2.5%, +1.1%]
→ ReLU wins even without SH
→ Conclusion: These tasks genuinely don't benefit from learned acts
```

**Outcome C: High variance**
```
Raw+ReLU:   R² = 0.720 ± 0.050
Raw+Spline: R² = 0.725 ± 0.055
CV > 50% for both
→ Raw coords fundamentally unstable
→ Explains why everyone uses SH/RFF encoding
```

---

## Experimental Design

### Same Structure as NB21/21b

**Experiments**:
1. Global Elevation (Raw, 10 seeds)
2. Global Population (Raw, 10 seeds)
3. Sample Size Sensitivity (5K to 50K)
4. Regional Multi-Seed (Himalayas/Sahara)
5. Extended Training (200 epochs)

**Activations**: ReLU, Spline, SIREN (skip RFF - known to work with raw but fails with SH)

**Statistics**: Mean±std, 95% CI, CV, paired t-tests (same as NB21/21b)

**Key difference**: `input_type='raw'` instead of `input_type='sh'`

---

## Decision Tree

### After NB21c Completes

**If Raw+Spline wins decisively (>2%, CV<20%)**:
1. ✅ Confirms simplicity paper predictions
2. ✅ SH was masking learned activation benefits
3. **Action**:
   - Investigate mechanistically: what content does SH remove?
   - Test if fine-tuning SH level (L=5, L=20) changes masking
   - Publication: "SH Pre-Encoding Masks Learned Activation Benefits"

**If Raw+ReLU still wins**:
1. ✅ Rules out "SH masking" hypothesis
2. ✅ Confirms these tasks don't benefit from learned activations
3. **Action**:
   - Accept that simplicity bias is sufficient for geographic tasks
   - Publication: "When Simplicity Bias Is Sufficient: Geographic Prediction"

**If Raw << SH (for both acts)**:
1. ✅ SH encoding is essential (provides +5-10% R²)
2. ✅ Justifies SH parameter cost
3. **Action**:
   - Continue with SH+ReLU as baseline
   - Accept NB21/21b conclusions about SH+acts performance

**If Raw shows high variance (CV>50%)**:
1. ⚠️ Raw coordinates fundamentally unstable
2. ⚠️ Explains prevalence of frequency encodings in literature
3. **Action**:
   - Stick with SH/RFF for stability
   - Publication: "Why Geographic Tasks Need Frequency Encodings"

---

## Expected Outcomes by Task

### Elevation (High-Frequency)

**Most likely**: Raw+Spline shows small advantage (+1-2%)
- Terrain has multi-scale structure
- Some high-frequency content remains without SH
- But variance may be higher than with SH

### Population (Smooth)

**Most likely**: Raw+ReLU wins or ties
- Population density is naturally smooth
- Little high-frequency content to capture
- Consistent with NB19b findings

---

## Computational Budget

**Total**: ~7 hours on Colab T4

**Breakdown**:
- Exp 1 (Elevation): 10 seeds × 3 acts × 90s = 45 min
- Exp 2 (Population): 10 seeds × 3 acts × 90s = 45 min
- Exp 3 (Sample Size): 4 sizes × 10 seeds × 2 acts × 90s = 3 hours
- Exp 4 (Regional): 2 regions × 2 sizes × 10 seeds × 2 acts × 100s = 2 hours
- Exp 5 (Convergence): 5 seeds × 2 acts × 180s = 30 min

**Efficiency**: Can run in parallel with NB21/21b analysis

---

## Why This Matters

### For Science

**Validates mechanistic hypothesis**:
- NB19/21 suggest learned activations don't help with SH
- But is this because SH masks their benefits?
- NB21c provides the control to answer this

**Aligns with simplicity paper**:
- Teney et al. used raw coordinates or learned embeddings
- Never tested with hand-crafted frequency encodings
- NB21c tests the key difference

### For Practice

**If Raw+Spline wins**:
- Developers should use raw+spline when SH not needed
- SH adds parameters but removes content

**If SH+acts >> Raw+acts**:
- Justifies SH parameter cost
- Explains why geographic tasks universally use encodings

**If Raw+ReLU wins**:
- Confirms simplicity bias sufficient
- Save compute by using simplest architecture

---

## Critical Success Metrics

**NB21c succeeds if it provides**:

1. ✅ **Clear variance estimates** for raw+acts (10 seeds)
2. ✅ **Direct comparison** to SH+acts (from NB21/21b)
3. ✅ **Statistical validation** of any differences (paired t-tests)
4. ✅ **Mechanistic insight**: Does SH mask or enhance learned act benefits?

**Even if results are "raw coords fail," NB21c succeeds if it clarifies WHY.**

---

## Integration with NB21/21b

### Parallel Validation

**All three notebooks test same question with different encodings**:
- NB21: SH+acts on elevation (multi-seed)
- NB21b: SH+acts on population (multi-seed)
- NB21c: Raw+acts on both tasks (multi-seed)

**Final synthesis** (after all complete):
```
                  Elevation               Population
           ReLU      Spline      ReLU      Spline
Raw      0.73±0.02  0.75±0.02   0.72±0.01  0.72±0.02
SH       0.90±0.01  0.90±0.01   0.74±0.01  0.74±0.01

Raw advantage:  +2.7%      +0%
SH value:      +23%       +3%
→ SH essential, masks small spline gain on elevation
```

### Cross-Validation

**Task-dependent effects**:
- If elevation and population differ → task properties matter
- If both show same pattern → generalizable finding

**Encoding-dependent effects**:
- If Raw and SH differ → encoding choice critical
- If same → encoding doesn't matter much

---

## Key Differences from NB21/21b

| Aspect | NB21/21b | NB21c |
|--------|----------|-------|
| **Input** | SH(L=10) = 121 dims | Raw = 2 dims |
| **Hypothesis** | Validate SH+acts reproducibility | Test if SH masks learned act gains |
| **Baseline** | SH+ReLU from NB19 | Raw+acts from NB14-16 |
| **Parameter count** | ~500K (3×256 MLP + SH encoder) | ~200K (3×256 MLP only) |
| **Expected R²** | 0.85-0.90 (elevation) | 0.70-0.75 (elevation) |
| **Key question** | "Are SH+acts results reproducible?" | "Does SH mask spline benefits?" |

---

## Related Work

### Teney et al. (Simplicity Paper)

**Key findings**:
- Learned activations help on high-frequency tasks
- Regression shows larger improvements than classification
- Effect size: 2-5% on synthetic tasks

**Difference**: They used raw coordinates, we use SH

**NB21c tests**: Are their findings hidden by our use of SH?

### Our NB19/20 Results

**With SH**:
- Spline gains tiny or negative
- ReLU wins most comparisons
- High variance on regional tasks

**NB21c will show**: Is this because SH pre-smooths signals?

---

## Bottom Line

**NB21c is the critical control experiment** to determine if:
1. SH encoding masks learned activation benefits (primary question)
2. Raw coordinates are viable without encodings (stability question)
3. Simplicity paper's findings apply to geographic tasks (validation question)

**After NB21c + NB21/21b complete**, we'll have:
- ✅ Reproducible baselines for both Raw and SH encodings
- ✅ Direct comparison to isolate encoding effects
- ✅ Clear guidance on when to use which activation + encoding
- ✅ Either strong positive result OR strong negative result (both publishable)

**This is highest-alpha work** - directly tests the mechanistic hypothesis all experts converged on.
